"""Inspect actual PDF text, raster canvases, SVG structure and plot alignment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
EXPORT=ROOT/'exports'
PROV=ROOT/'provenance'
NAMES=[f'Figure_{i}' for i in range(1,6)]+[f'Figure_S{i}' for i in range(1,8)]


def main():
    html=shutil.which('pdftohtml');bbox=shutil.which('pdftotext');assert html and bbox
    report={'figures':{},'rendering_backend':'R 4.6.1 base graphics; 400 dpi Poppler rasterization of canonical R PDFs',
            'all_text_black':True,'all_canvases_white':True,'scientific_reanalysis_executed':False}
    for name in NAMES:
        pdf=EXPORT/(name+'.pdf');xml=subprocess.run([html,'-xml','-i','-hidden','-stdout',str(pdf)],capture_output=True,check=True).stdout
        # Poppler sometimes includes diagnostic prose before the XML declaration.
        start=xml.find(b'<?xml');xml=xml[start:]
        tree=ET.fromstring(xml);pages=tree.findall('.//page');assert len(pages)==1
        fonts=tree.findall('.//fontspec');colors={f.attrib.get('color','').lower() for f in fonts};assert colors and colors<={'#000000'},(name,colors)
        page=pages[0];w=float(page.attrib['width']);h=float(page.attrib['height']);outside=[]
        for text in tree.findall('.//text'):
            x,y=float(text.attrib['left']),float(text.attrib['top']);rw,rh=float(text.attrib['width']),float(text.attrib['height'])
            if x<-.5 or y<-.5 or x+rw>w+.5 or y+rh>h+.5:outside.append({'text':''.join(text.itertext()),'box':[x,y,rw,rh]})
        assert not outside,(name,outside)
        headers=list(csv.DictReader((PROV/(name+'_headers.csv')).open()))
        assert len(headers)==(6 if name in ('Figure_2','Figure_4','Figure_5','Figure_S4') else 4)
        # Text extraction verifies the visible panel navigation, not just code intent.
        extracted=[''.join(t.itertext()).strip() for t in tree.findall('.//text')]
        assert all(any(row['letter']==text for text in extracted) for row in headers),(name,'missing visible panel letter',extracted[:8])
        cols=3 if len(headers)==6 else 2
        aligned=[]
        for column in range(cols):
            candidates=[r for idx,r in enumerate(headers) if idx%cols==column and not(name=='Figure_1' and r['letter']=='a')]
            if len(candidates)>1:
                for edge in ('plot_left','plot_right'):
                    values=[float(r[edge]) for r in candidates];assert max(values)-min(values)<1e-7,(name,column,edge,values)
                aligned.append(column+1)
        for rowidx in (0,1):
            candidates=[r for idx,r in enumerate(headers) if idx//cols==rowidx and not(name=='Figure_1' and r['letter']=='a')]
            if len(candidates)>1:
                for edge in ('plot_bottom','plot_top'):
                    values=[float(r[edge]) for r in candidates];assert max(values)-min(values)<1e-7,(name,rowidx,edge,values)
        image=Image.open(EXPORT/(name+'.png')).convert('RGB');iw,ih=image.size
        assert all(image.getpixel(point)==(255,255,255) for point in ((0,0),(iw-1,0),(0,ih-1),(iw-1,ih-1))),name
        svg=ET.parse(EXPORT/(name+'.svg'));assert svg.getroot().tag.endswith('svg')
        glyphs=[]
        def verify_glyph_colours(node,inherited='black'):
            style=dict(part.split(':',1) for part in node.attrib.get('style','').split(';') if ':' in part)
            fill=style.get('fill',node.attrib.get('fill',inherited)).replace(' ','').lower()
            href=node.attrib.get('{http://www.w3.org/1999/xlink}href','')
            if node.tag.endswith('use') and href.startswith('#glyph'):
                assert fill in ('black','#000000','#000','rgb(0%,0%,0%)','rgb(0,0,0)'),(name,href,fill)
                glyphs.append(href)
            for child in node:verify_glyph_colours(child,fill)
        verify_glyph_colours(svg.getroot());assert glyphs
        report['figures'][name]={'pdf_pages':1,'text_colors':sorted(colors),'minimum_font_size_poppler_units':min(float(f.attrib['size']) for f in fonts),
            'text_outside_page':outside,'panel_labels_verified':len(headers),'aligned_columns_verified':aligned,
            'png_dimensions':[iw,ih],'white_canvas_verified':True,'svg_black_text_glyphs_verified':len(glyphs),
            'sha256':{ext:hashlib.sha256((EXPORT/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ('pdf','svg','png')}}
    for bundle,expected in (('Main_Figures',5),('Supplementary_Figures',7)):
        xml=subprocess.run([html,'-xml','-i','-stdout',str(EXPORT/(bundle+'.pdf'))],capture_output=True,check=True).stdout
        tree=ET.fromstring(xml[xml.find(b'<?xml'):]);assert len(tree.findall('.//page'))==expected
    # Contact sheets are QA previews only; scientific panels themselves are drawn in R.
    for label,names in (('Main',NAMES[:5]),('SI',NAMES[5:])):
        thumbw,thumbh=740,640;columns=2;rows=(len(names)+1)//2
        sheet=Image.new('RGB',(columns*thumbw,rows*(thumbh+35)),'white');draw=ImageDraw.Draw(sheet)
        for idx,name in enumerate(names):
            image=Image.open(EXPORT/(name+'.png')).convert('RGB');image.thumbnail((thumbw,thumbh))
            x=(idx%columns)*thumbw;y=(idx//columns)*(thumbh+35);sheet.paste(image,(x,y+30));draw.text((x+16,y+8),name,fill='black')
        sheet.save(PROV/(label+'_contact_sheet.png'))
    (PROV/'export_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'figures_verified':len(NAMES),'pdf_text_black':True,'white_backgrounds':True,'panel_alignment_verified':True,'bundle_pages':[5,7]}))


if __name__=='__main__':main()
