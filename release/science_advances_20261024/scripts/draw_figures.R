# Observation constraints on mechanistic inference.
# All scientific artwork is rendered in R. No model fitting or significance tests.
options(stringsAsFactors=FALSE, warn=1)
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) normalizePath(args[1],winslash="/") else normalizePath("figures/mechanistic_inference_20261024",winslash="/")
data_dir <- file.path(root,"data"); plot_dir <- file.path(root,"plot_data")
out <- file.path(root,"exports"); dir.create(out,showWarnings=FALSE)
P <- c(paper="#f5f2ec",neutral="#d8d4cd",ink="#756458",accent="#b54a3c")
CANVAS <- "#ffffff"
TEXT <- "#000000"
mc <- c(training_mean=P["neutral"],cortical=P["ink"],effective_mode=P["ink"],full_population=P["accent"],
        observed_input_only=P["neutral"],factors_and_observed_input=P["ink"],cortical_and_observed_input=P["ink"],
        effective_mode_and_observed_input=P["ink"],factors_effective_mode_and_observed_input=P["accent"],full_population_and_observed_input=P["accent"])
names(mc) <- c("training_mean","cortical","effective_mode","full_population","observed_input_only","factors_and_observed_input", "cortical_and_observed_input","effective_mode_and_observed_input","factors_effective_mode_and_observed_input","full_population_and_observed_input")
ml <- c(training_mean="Mean",cortical="Cortical",effective_mode="Mode",full_population="Population",observed_input_only="Input",factors_and_observed_input="Factors + u",cortical_and_observed_input="Cortex + u",effective_mode_and_observed_input="Mode + u",factors_effective_mode_and_observed_input="Factors + mode + u",full_population_and_observed_input="Population + u")
reps <- c("cortical","effective_mode","full_population")
roles <- c("train","validation","test"); role_shapes <- c(train=16,validation=17,test=1)
family_shapes <- c(gain=16,coupling=17,history=15,combined=1,published_bank=16,published_reference=17,runtime_diagnostic=3)
read <- function(path,derived=FALSE) read.csv(file.path(if(derived) plot_dir else data_dir,path),check.names=FALSE,na.strings=c("","nan","NaN","None"))
truth <- function(x) as.character(x)%in%c("True","TRUE","true")
fade <- function(col,alpha=.4) adjustcolor(unname(col),alpha.f=alpha)
range_safe <- function(x,zero=FALSE,pad=.08) {r<-range(x[is.finite(x)]);if(zero)r<-range(c(0,r));if(diff(r)==0)r<-r+c(-.5,.5);r+c(-1,1)*diff(r)*pad}
q <- function(x) quantile(x[is.finite(x)],c(.25,.5,.75),names=FALSE)
family_label <- function(x) c(gain="Gain",coupling="Coupling",history="History",combined="Combined",published_bank="Published",published_reference="Reference",runtime_diagnostic="Diagnostic")[x]
HEADERS <- list()
reset <- function(nr=2,nc=2,mar=c(3.8,5.8,2.6,.9)) {
  HEADERS <<- list()
  par(mfrow=c(nr,nc),mar=mar,oma=c(.3,.4,.3,.2),family="Arial",fg=P["ink"],col=TEXT,col.axis=TEXT,col.lab=TEXT,col.main=TEXT,
      cex=.88,cex.axis=.83,cex.lab=.9,tcl=-.2,mgp=c(2.1,.48,0),las=1,lwd=.8,xaxs="r",yaxs="r",bg=CANVAS)
}
head_panel <- function(letter,title) {
  fig<-par("fig");dx<-diff(fig[1:2]);dy<-diff(fig[3:4]);y<-fig[4]-.045*dy
  plt<-par("plt")
  HEADERS[[length(HEADERS)+1]] <<- list(letter=letter,title=title,x=fig[1]+c(.035,.11)*dx,y=y,
    left=fig[1]+dx*plt[1],right=fig[1]+dx*plt[2],bottom=fig[3]+dy*plt[3],top=fig[3]+dy*plt[4])
}
render_figure <- function(fun) {
  fun()
  par(fig=c(0,1,0,1),mar=c(0,0,0,0),new=TRUE,xpd=NA,cex=1,ps=10.5,family="Arial",col=TEXT)
  plot.new();plot.window(c(0,1),c(0,1),xaxs="i",yaxs="i",log="")
  for(h in HEADERS) {
    text(h$x[1],h$y,h$letter,adj=c(0,1),cex=10.3/10.5,font=2,col=TEXT)
    text(h$x[2],h$y,h$title,adj=c(0,1),cex=8.1/10.5,col=TEXT)
  }
}
axes <- function(x=TRUE,y=TRUE) {if(x)axis(1,col=P["ink"],col.ticks=P["ink"]);if(y)axis(2,col=P["ink"],col.ticks=P["ink"]);box(bty="l",col=P["ink"])}
panel <- function(letter,title,xlab="",ylab="",xlim=c(0,1),ylim=c(0,1),log="",xaxis=TRUE,yaxis=TRUE) {
  plot.new();plot.window(xlim=xlim,ylim=ylim,log=log);axes(xaxis,yaxis);title(xlab=xlab,ylab=ylab);head_panel(letter,title)
}
key <- function(labels,col,pch=16,pos="topright",ncol=1) legend(pos,legend=labels,col=unname(col),text.col=TEXT,pch=pch,bty="n",cex=.76,ncol=ncol,x.intersp=.65,y.intersp=.95,inset=.012)
zero <- function(h=NULL,v=NULL) {if(!is.null(h))abline(h=h,col=fade(P["ink"],.5),lty=3);if(!is.null(v))abline(v=v,col=fade(P["ink"],.5),lty=3)}
identity <- function()abline(0,1,col=fade(P["ink"],.6),lty=3)
forest <- function(d,letter,title,estimate="mean_standardized_mse",lo="lower",hi="upper",labels=NULL,xlab="Standardized MSE",at_zero=FALSE) {
  if(is.null(labels)) labels<-unname(ml[d$model])
  lim<-range_safe(c(d[[lo]],d[[hi]],d[[estimate]]),zero=TRUE)
  panel(letter,title,xlab,"",lim,c(.5,nrow(d)+.5),yaxis=FALSE);axis(2,at=rev(seq_len(nrow(d))),labels=labels,las=1,tick=FALSE,cex.axis=.79)
  if(at_zero)zero(v=0)
  for(i in seq_len(nrow(d))) {y<-nrow(d)+1-i;cl<-if("model"%in%names(d))mc[d$model[i]] else P["accent"]
    segments(d[[lo]][i],y,d[[hi]][i],y,col=cl,lwd=1.7);points(d[[estimate]][i],y,pch=21,bg=cl,col=P["ink"],cex=.9)
  }
}
calibration <- function(d,target,models,letter,title,xlab,ylab="Predicted") {
  d<-d[d$model%in%models,]; xx<-d[[paste0("true_",target)]];yy<-d[[paste0("predicted_",target)]];lim<-range_safe(c(xx,yy),TRUE)
  panel(letter,title,xlab,ylab,lim,lim);identity()
  for(j in seq_along(models)) {r<-d[d$model==models[j],];points(r[[paste0("true_",target)]],r[[paste0("predicted_",target)]],pch=c(1,16,17,15)[j],col=mc[models[j]],cex=.84)}
}
paired <- function(d,model_a,model_b,letter,title,labels=c("Cortical","Population"),metric="standardized_mse",ylab="Standardized MSE") {
  aa<-d[d$model==model_a,c("configuration_group",metric)];bb<-d[d$model==model_b,c("configuration_group",metric)];z<-merge(aa,bb,by="configuration_group",suffixes=c("_a","_b"));stopifnot(nrow(z)==10)
  a<-z[[paste0(metric,"_a")]];b<-z[[paste0(metric,"_b")]]
  panel(letter,title,"",ylab,c(.7,2.3),range_safe(c(a,b),TRUE),xaxis=FALSE);axis(1,at=1:2,labels=labels,tick=FALSE)
  for(i in seq_along(a))lines(1:2,c(a[i],b[i]),col=fade(P["ink"],.28),lwd=.75)
  points(rep(1,length(a)),a,pch=1,col=P["ink"],cex=.75);points(rep(2,length(b)),b,pch=16,col=P["accent"],cex=.75)
  points(1:2,c(mean(a),mean(b)),pch=23,bg=c(P["ink"],P["accent"]),col=P["paper"],cex=1.4)
}
boxpoints <- function(values,groups,levels,cols,letter,title,ylab,ylim=NULL,labels=levels,log="") {
  f<-factor(groups,levels=levels);keep<-is.finite(values)&!is.na(f);values<-values[keep];f<-f[keep]
  if(is.null(ylim))ylim<-range_safe(values)
  panel(letter,title,"",ylab,c(.45,length(levels)+.55),ylim,log=log,xaxis=FALSE)
  boxplot(split(values,f),at=seq_along(levels),add=TRUE,axes=FALSE,outline=FALSE,boxwex=.45,col=ifelse(cols==P["accent"],fade(P["accent"],.14),P["paper"]),border=cols,medlwd=1.4,staplelty=0,whisklty=1)
  for(i in seq_along(levels)) {v<-values[f==levels[i]];j<-if(length(v)>1)seq(-.16,.16,length.out=length(v)) else 0;points(i+j,v,pch=16,col=fade(cols[i],.35),cex=.45)}
  axis(1,at=seq_along(levels),labels=labels,tick=FALSE,cex.axis=.78)
}
heat <- function(mat,letter,title,xlabels=colnames(mat),ylabels=rownames(mat),zlim=NULL,symmetric=FALSE,xlab="",ylab="") {
  if(is.null(zlim))zlim<-range(mat[is.finite(mat)]);if(symmetric)zlim<-c(-1,1)*max(abs(zlim));if(diff(zlim)==0)zlim<-zlim+c(-.5,.5)
  ramp<-colorRampPalette(if(symmetric)c(P["ink"],P["paper"],P["accent"]) else c(P["paper"],P["neutral"],P["ink"],P["accent"]))(128)
  panel(letter,title,xlab,ylab,c(.5,ncol(mat)+.5),c(.5,nrow(mat)+.5),xaxis=FALSE,yaxis=FALSE)
  for(i in seq_len(nrow(mat)))for(j in seq_len(ncol(mat))) {v<-mat[i,j];cl<-if(is.finite(v))ramp[1+round(127*(max(zlim[1],min(zlim[2],v))-zlim[1])/diff(zlim))] else P["neutral"]
    rect(j-.48,nrow(mat)-i+.52,j+.48,nrow(mat)-i+1.48,col=cl,border=NA)}
  axis(1,at=seq_len(ncol(mat)),labels=xlabels,tick=FALSE,cex.axis=.72);axis(2,at=rev(seq_len(nrow(mat))),labels=ylabels,tick=FALSE,cex.axis=.72)
  mtext(paste(format(zlim[1],digits=2),"  -  ",format(zlim[2],digits=2)),side=3,line=.12,adj=1,cex=.68)
}
factors<-read("factor_roles.csv",TRUE);features<-read("mechanism_compensation/zero_input_features.csv");rp<-read("mechanism_compensation/parameter_predictions.csv");pp<-read("mechanism_compensation/finite_response_predictions.csv");near<-read("mechanism_compensation/cross_family_neighbours.csv")
rs<-read("recovery_summary.csv",TRUE);ps<-read("response_summary.csv",TRUE);vg<-read("validation_grid.csv",TRUE);cx<-read("model_complexity.csv",TRUE);sv<-read("sensitivity_singular_values.csv",TRUE)
balance<-read("pathway_balance/population_pathway_balance.csv");edge<-read("pathway_balance/edge_response_timing.csv");pc<-read("pathway_balance/configuration_summary.csv");matches<-read("pathway_balance/matched_pathway_contrasts.csv")
human<-read("human_comparisons.csv",TRUE);he<-read("human_person_effects.csv",TRUE);pred<-read("human_spatial/held_person_predictions.csv");ef<-read("human_spatial/episode_features.csv");disp<-read("human_spatial/episode_dispositions.csv")
stopifnot(nrow(features)==616,nrow(rp)==40,nrow(pp)==70,nrow(balance)==7524,nrow(edge)==1968,nrow(factors)==48)

fig1 <- function() {
  reset(2,2)
  par(mar=c(1.1,1,2.6,.5));plot.new();plot.window(c(-1.6,1.7),c(-1.25,1.35));head_panel("a","Native populations")
  ang<-pi/2-2*pi*(0:8)/9;xx<-cos(ang);yy<-sin(ang);xx<-c(xx,-1.4,1.4);yy<-c(yy,-.85,-.85)
  con<-edge[!duplicated(edge$connection_id),];for(i in seq_len(nrow(con))) {a<-con$source_population[i];b<-con$destination_population[i];if(a!=b)arrows(xx[a],yy[a],xx[b],yy[b],length=.04,col=fade(P["ink"],.25),lwd=.7)}
  symbols(xx[1:9],yy[1:9],circles=rep(.105,9),inches=FALSE,add=TRUE,bg=c(P["accent"],rep(P["neutral"],8)),fg=P["ink"])
  text(xx[1:9],yy[1:9],1:9,col=TEXT,cex=.82)
  rect(xx[10:11]-.14,yy[10:11]-.1,xx[10:11]+.14,yy[10:11]+.1,col=P["paper"],border=P["ink"]);text(xx[10:11],yy[10:11],c("noise","u"),cex=.7)
  text(0,-1.22,"1 spatial node",cex=.76)
  par(mar=c(3.8,5.8,2.6,.9));panel("b","Recorded bank","","Native records",c(.4,3.6),c(0,400),xaxis=FALSE)
  vals<-c(284,332,48);cols<-c(P["ink"],P["accent"],P["neutral"]);rect(1:3-.29,0,1:3+.29,vals,col=cols,border=NA);text(1:3,vals+25,vals,cex=.85)
  axis(1,at=1:3,labels=c("Pristine\nzero input","Patched\nzero input","Patched\npulse"),tick=FALSE,cex.axis=.78)
  panel("c","Manipulated configurations","Factor multiplier","",c(.42,2.25),c(.5,4.2),log="x",yaxis=FALSE)
  axis(2,at=1:3,labels=c("Gain","Coupling","History"),las=1,tick=FALSE)
  for(j in 1:3) {v<-factors[[c("param_gain_factor","param_coupling_factor","param_history_factor")[j]]];jit<-seq(-.15,.15,length.out=nrow(factors));points(v,j+jit,pch=unname(role_shapes[factors$role]),col=ifelse(factors$role=="test",P["accent"],P["ink"]),cex=.64)}
  abline(v=1,lty=3,col=P["neutral"]);key(c("Train (28)","Validation (10)","Test (10)"),c(P["ink"],P["ink"],P["accent"]),c(16,17,1),pos="topright")
  panel("d","Recorded pulse protocol","Time relative to output start (s)","Input amplitude",c(0,10),c(-.1,1.5),yaxis=FALSE)
  axis(2,at=c(0,1),labels=c("0","A"),las=1);lines(c(.002,1.998,2,2.02,2.022,10),c(0,0,1,1,0,0),col=P["accent"],lwd=1.7)
  arrows(2,1.24,2,1.05,length=.055,col=P["ink"]);text(2.1,1.35,"20 ms",adj=0,cex=.78);text(7,1.15,"A = 1, 10, 100",cex=.78);text(7,.9,"48 matched pairs",cex=.78)
}

fig2 <- function() {
  reset(2,3)
  ord<-c("training_mean","cortical","effective_mode","full_population");d<-rs[match(ord,rs$model),];forest(d,"a","Parameter recovery")
  paired(rp,"cortical","full_population","b","Paired configuration loss")
  calibration(rp,"log_gain",c("cortical","full_population"),"c","Synaptic gain","True log multiplier");key(c("Cortical","Population"),c(P["ink"],P["accent"]),c(1,16),pos="topleft")
  calibration(rp,"log_coupling",c("cortical","full_population"),"d","Coupling","True log multiplier")
  calibration(rp,"log_history",c("cortical","full_population"),"e","Synaptic history","True log multiplier")
  z<-cx[cx$task=="parameter_recovery",];z<-merge(z,rs[,c("model","mean_standardized_mse")],by="model")
  panel("f","Fitted capacity","Input feature count","Effective degrees of freedom",c(-3,65),c(0,15))
  for(i in seq_len(nrow(z))) {points(z$input_columns[i],z$effective_df[i],pch=21,col=P["ink"],bg=mc[z$model[i]],cex=1.05);text(z$input_columns[i]+if(z$model[i]=="full_population")-2 else 2,z$effective_df[i],unname(ml[z$model[i]]),adj=if(z$model[i]=="full_population")1 else 0,cex=.75)}
}

fig3 <- function() {
  reset(2,2)
  panel("a","Population-state dimensionality","Voltage participation ratio","Rate participation ratio",range_safe(features$V_raw_participation),range_safe(features$Q_raw_participation))
  for(v in c("pristine","scaled_dendrite_v1_explicit_patch")) {d<-features[features$variant==v,];points(d$V_raw_participation,d$Q_raw_participation,pch=if(v=="pristine")1 else 16,col=fade(if(v=="pristine")P["ink"] else P["accent"],.45),cex=.55)}
  key(c("Pristine (284)","Patched (332)"),c(P["ink"],P["accent"]),c(1,16),pos="topleft")
  panel("b","Training feature sensitivity","Singular component","Singular value",c(.8,3.2),c(0,.75),xaxis=FALSE);axis(1,at=1:3)
  for(j in seq_along(reps)) {d<-sv[sv$representation==reps[j],];lines(d$component,d$singular_value,col=mc[reps[j]],lty=j,pch=c(1,17,16)[j],type="b",cex=.75,lwd=1.3)}
  legend("topright",legend=unname(ml[reps]),col=unname(mc[reps]),text.col=TEXT,lty=1:3,pch=c(1,17,16),bty="n",cex=.76)
  n<-near[near$matching=="different_training_family",];xlim<-range_safe(n$feature_rms_distance,TRUE);ylim<-range_safe(n$log_factor_rms_distance,TRUE)
  panel("c","Cross-family neighbours","Training-scaled feature distance","Log-factor RMS distance",xlim,ylim)
  for(j in seq_along(reps)){d<-n[n$representation==reps[j],];points(d$feature_rms_distance,d$log_factor_rms_distance,pch=c(1,17,16)[j],col=mc[reps[j]],cex=.8)}
  key(unname(ml[reps]),mc[reps],c(1,17,16),pos="topright")
  panel("d","Gain-coupling proximity","Log-factor RMS distance","Absolute log(gain x coupling) distance",range_safe(n$log_factor_rms_distance,TRUE),range_safe(n$log_gain_plus_coupling_distance,TRUE))
  for(j in seq_along(reps)){d<-n[n$representation==reps[j],];points(d$log_factor_rms_distance,d$log_gain_plus_coupling_distance,pch=c(1,17,16)[j],col=mc[reps[j]],cex=.8)}
}

fig4 <- function() {
  reset(2,3)
  ord<-c("training_mean","observed_input_only","factors_and_observed_input","cortical_and_observed_input","effective_mode_and_observed_input","factors_effective_mode_and_observed_input","full_population_and_observed_input")
  d<-ps[match(ord,ps$model),];short<-c("Mean","Input","Factors + u","Cortex + u","Mode + u","Combined","Population + u")
  forest(d,"a","Response prediction",labels=short)
  contrast_models<-c("observed_input_only","cortical_and_observed_input","effective_mode_and_observed_input","factors_effective_mode_and_observed_input")
  db<-d[match(contrast_models,d$model),]
  forest(db,"b","Summary-feature contrasts",estimate="loss_minus_baseline",lo="difference_lower",hi="difference_upper",labels=c("Input","Cortex + u","Mode + u","Combined"),xlab="MSE difference",at_zero=TRUE)
  paired(pp,"factors_and_observed_input","factors_effective_mode_and_observed_input","c","Paired response losses",labels=c("Factors + u","Combined"))
  calibration(pp,"voltage_peak_rms",c("factors_and_observed_input","factors_effective_mode_and_observed_input"),"d","Peak voltage response","Observed RMS peak (V)","Predicted RMS peak (V)")
  key(c("Factors + u","Combined"),c(P["ink"],P["accent"]),c(1,16),pos="topleft")
  calibration(pp,"rate_peak_rms",c("factors_and_observed_input","factors_effective_mode_and_observed_input"),"e","Peak firing-rate response","Observed RMS peak (s^-1)","Predicted RMS peak (s^-1)")
  selected<-c("factors_and_observed_input","effective_mode_and_observed_input","factors_effective_mode_and_observed_input","full_population_and_observed_input");z<-pp[pp$model%in%selected,]
  panel("f","Observed input strata","Pulse amplitude","Standardized MSE",c(.7,145),range_safe(z$standardized_mse,TRUE),log="x",xaxis=FALSE);axis(1,at=c(1,10,100))
  for(j in seq_along(selected)){d<-z[z$model==selected[j],];points(d$observed_input_amplitude*10^((j-2.5)*.035),d$standardized_mse,pch=c(1,17,16,15)[j],col=fade(mc[selected[j]],.8),cex=.68)}
  key(c("Factors + u","Mode + u","Combined","Population + u"),mc[selected],c(1,17,16,15),pos="topright")
}

fig5 <- function() {
  reset(2,3)
  pulse<-balance[balance$mode=="paired_finite_pulse_response"&balance$scope=="all_incoming",];pulse<-pulse[is.finite(pulse$absolute_pathway_mass_V_s)&pulse$absolute_pathway_mass_V_s>0&pulse$net_pathway_mass_V_s>0,]
  lim<-range_safe(log10(c(pulse$absolute_pathway_mass_V_s,pulse$net_pathway_mass_V_s)))
  panel("a","Pulse pathway mass","log10 absolute mass (V s)","log10 net mass (V s)",lim,lim);identity()
  points(log10(pulse$absolute_pathway_mass_V_s),log10(pulse$net_pathway_mass_V_s),pch=ifelse(pulse$population==1,16,1),col=ifelse(pulse$population==1,fade(P["accent"],.7),fade(P["ink"],.3)),cex=.52)
  key(c("Population 1","Populations 2-9"),c(P["accent"],P["ink"]),c(16,1),pos="topleft")
  boxpoints(pulse$cancellation_for_inference,pulse$population,1:9,rep(P["accent"],9),"b","Pulse cancellation","Voltage cancellation",c(0,1),labels=1:9);mtext("Population",side=1,line=2.1,cex=.9)
  z<-balance[balance$mode=="zero_input_centered_fluctuation"&balance$scope=="all_incoming",]
  boxpoints(z$cancellation_for_inference,z$population,1:9,rep(P["ink"],9),"c","Zero-input fluctuations","Voltage cancellation",c(0,1),labels=1:9);mtext("Population",side=1,line=2.1,cex=.9)
  hidden<-aggregate(cancellation_for_inference~configuration_cluster,pulse[pulse$population!=1,],mean);cort<-pulse[pulse$population==1,c("configuration_cluster","cancellation_for_inference")];z<-merge(cort,hidden,by="configuration_cluster",suffixes=c("_cort","_hidden"))
  panel("d","Population cancellation","Population 1 cancellation","Other-population mean cancellation",c(0,1),c(0,1));identity();points(z$cancellation_for_inference_cort,z$cancellation_for_inference_hidden,pch=16,col=fade(P["accent"],.65),cex=.65)
  panel("e","Matching coverage","","Matched configuration pairs",c(.7,3.3),c(0,3.3),xaxis=FALSE,yaxis=FALSE);axis(1,at=1:3,labels=c("Score","Features","Fit"),tick=FALSE,cex.axis=.75);axis(2,at=log10(c(0,1,4,10,100,1000)+1),labels=c(0,1,4,10,100,1000),las=1)
  for(j in 1:2){v<-list(c(1014,4,4),c(168,1,0))[[j]];y<-log10(v+1);x<-1:3+c(-.05,.05)[j];lines(x,y,type="b",pch=c(1,16)[j],col=c(P["ink"],P["accent"])[j],lwd=1.2);text(x,y+.12,as.character(v),cex=.7,col=TEXT)}
  key(c("Zero input","Pulse"),c(P["ink"],P["accent"]),c(1,16),pos="topright")
  panel("f","Raw signature distances","Population RMS difference (mV)","Pathway RMS difference (mV)",c(0,17.5),c(0,8))
  points(1000*matches$raw_population_signature_rms_difference_V,1000*matches$raw_pathway_signature_rms_difference_V,pch=16,col=fade(P["ink"],.14),cex=.42)
  z<-matches[matches$matching=="score_and_conventional_features",];points(1000*z$raw_population_signature_rms_difference_V,1000*z$raw_pathway_signature_rms_difference_V,pch=21,bg=P["accent"],col=P["paper"],cex=1)
  key(c("Score-matched","Feature-matched"),c(P["ink"],P["accent"]),c(16,16),pos="topright")
}

s1 <- function() {
  reset(2,2)
  pairs<-list(c("gain","coupling"),c("gain","history"),c("coupling","history"))
  for(k in 1:3){a<-pairs[[k]][1];b<-pairs[[k]][2];panel(letters[k],paste(tools::toTitleCase(a),"and",b),paste(a,"multiplier"),paste(b,"multiplier"),c(.45,2.15),c(.45,2.15));zero(v=1,h=1)
    for(r in roles){z<-factors[factors$role==r,];points(z[[paste0("param_",a,"_factor")]],z[[paste0("param_",b,"_factor")]],pch=role_shapes[r],col=if(r=="test")P["accent"] else P["ink"],cex=.8)}
    if(k==1)key(c("Train","Validation","Test"),c(P["ink"],P["ink"],P["accent"]),c(16,17,1),pos="topleft")}
  tab<-table(factor(factors$account_class,levels=c("gain","coupling","history","combined")),factor(factors$role,levels=roles));bp<-barplot(t(tab),beside=TRUE,col=c(P["neutral"],P["ink"],P["accent"]),border=NA,names.arg=c("Gain","Coupling","History","Combined"),ylim=c(0,max(tab)+4),ylab="Configuration groups",axes=FALSE);axis(2,las=1);box(bty="l");head_panel("d","Family and split census");legend("topleft",legend=c("Train","Validation","Test"),fill=c(P["neutral"],P["ink"],P["accent"]),text.col=TEXT,border=NA,bty="n",cex=.78);text(bp,as.vector(t(tab))+.65,as.vector(t(tab)),cex=.72)
}

s2 <- function() {
  reset(2,2)
  specs<-list(c("parameter_recovery","validation_standardized_mse"),c("parameter_recovery","effective_df"),c("finite_response_prediction","validation_standardized_mse"),c("finite_response_prediction","effective_df"))
  for(k in 1:4){task<-specs[[k]][1];metric<-specs[[k]][2];d<-vg[vg$task==task,];mods<-unique(d$model);panel(letters[k],paste(if(task=="parameter_recovery")"Recovery" else "Pulse",if(metric=="effective_df")"capacity" else "validation"),"Ridge penalty",if(metric=="effective_df")"Effective degrees of freedom" else "Validation standardized MSE",c(.007,150),range_safe(d[[metric]],TRUE),log="x",xaxis=FALSE);axis(1,at=c(.01,1,100),labels=c("0.01","1","100"))
    for(j in seq_along(mods)){z<-d[d$model==mods[j],];lines(z$ridge,z[[metric]],type="b",pch=c(1,17,16,15,2,0)[j],lty=1+(j%%3),col=mc[mods[j]],cex=.7,lwd=1.1)}
    if(k%%2==1)legend("topright",legend=unname(ml[mods]),col=unname(mc[mods]),text.col=TEXT,pch=c(1,17,16,15,2,0)[seq_along(mods)],lty=1+(seq_along(mods)%%3),bty="n",cex=.7)}
}

s3 <- function() {
  reset(2,2)
  for(k in 1:3){target<-c("gain","coupling","history")[k];d<-rp[rp$model%in%c("cortical","full_population"),];res<-d[[paste0("predicted_log_",target)]]-d[[paste0("true_log_",target)]];panel(letters[k],paste(tools::toTitleCase(target),"residuals"),"True log multiplier","Predicted minus true",range_safe(d[[paste0("true_log_",target)]]),range_safe(res,TRUE));zero(h=0)
    points(d[[paste0("true_log_",target)]],res,pch=ifelse(d$model=="cortical",1,16),col=ifelse(d$model=="cortical",P["ink"],P["accent"]),cex=.8);if(k==1)key(c("Cortical","Population"),c(P["ink"],P["accent"]),c(1,16))}
  aa<-rp[rp$model=="cortical",c("configuration_group","account_class","standardized_mse")];bb<-rp[rp$model=="full_population",c("configuration_group","standardized_mse")];z<-merge(aa,bb,by="configuration_group",suffixes=c("_cort","_pop"));delta<-z$standardized_mse_pop-z$standardized_mse_cort
  boxpoints(delta,z$account_class,c("gain","coupling","history","combined"),c(P["ink"],P["ink"],P["ink"],P["accent"]),"d","Paired loss by parameter family","Population minus cortical MSE",ylim=range_safe(delta,TRUE),labels=c("Gain","Coupling","History","Combined"));zero(h=0)
}

s4 <- function() {
  reset(2,3)
  targets<-c("voltage_peak_rms","rate_peak_rms","voltage_integrated_rms","rate_integrated_rms","voltage_mode_fraction","rate_mode_fraction")
  labels<-c("Voltage peak (V)","Rate peak (s^-1)","Voltage integral (V s)","Rate integral","Voltage mode fraction","Rate mode fraction")
  for(k in 1:6){calibration(pp,targets[k],c("factors_and_observed_input","factors_effective_mode_and_observed_input"),letters[k],labels[k],"Observed","Predicted");if(k==1)key(c("Factors + u","Combined"),c(P["ink"],P["accent"]),c(1,16),pos="topleft")}
}

s5 <- function() {
  reset(2,2)
  contrasts<-c("experience_presence","content_recall_given_experience")
  for(k in 1:2){z<-pred[pred$panel=="primary_common_average"&pred$contrast==contrasts[k],];a<-aggregate(cbind(baseline_brier,spatial_brier)~participant_id,z,mean)
    panel(letters[k],if(k==1)"Experience presence" else "Content recall","","Person-mean Brier loss",c(.7,2.3),range_safe(c(a$baseline_brier,a$spatial_brier),TRUE),xaxis=FALSE);axis(1,at=1:2,labels=c("Score + global","+ posterior"),tick=FALSE,cex.axis=.76)
    for(i in seq_len(nrow(a)))lines(1:2,c(a$baseline_brier[i],a$spatial_brier[i]),col=fade(P["ink"],.4));points(rep(1,nrow(a)),a$baseline_brier,pch=1,col=P["ink"]);points(rep(2,nrow(a)),a$spatial_brier,pch=16,col=P["accent"])}
  for(k in 1:2){z<-he[he$panel=="primary_common_average"&he$contrast==contrasts[k]&he$kind=="all_people_retrospective_within_person_matched_contrast",];z<-z[order(z$difference),];panel(letters[k+2],if(k==1)"Matched experience contrast" else "Matched recall contrast","Person-mean log10 power-ratio difference","Participant",range_safe(z$difference,TRUE),c(.5,nrow(z)+.5),yaxis=FALSE);zero(v=0);axis(2,at=seq_len(nrow(z)),labels=sub("dream_tononi_serial_","",z$participant_id),las=1,tick=FALSE,cex.axis=.7);points(z$difference,seq_len(nrow(z)),pch=16,col=P["accent"],cex=.75)}
}

s6 <- function() {
  reset(2,2,mar=c(3.8,9,2.6,.9))
  lines<-readLines(file.path(data_dir,"electrode_coordinates.sfp"));coords<-do.call(rbind,lapply(lines,function(line){tok<-strsplit(trimws(line),"[[:space:]]+")[[1]];if(length(tok)==4&&grepl("^[0-9]+$",tok[1]))as.numeric(tok) else NULL}));coords<-coords[coords[,4]>=0,]
  panel("a","Fixed sensor regions","x (cm)","y (cm)",range_safe(coords[,2]),range_safe(coords[,3]));reg<-ifelse(coords[,3]<=-3,"Posterior",ifelse(coords[,3]>=3,"Frontal","Other"));cl<-ifelse(reg=="Posterior",P["accent"],ifelse(reg=="Frontal",P["ink"],P["neutral"]));points(coords[,2],coords[,3],pch=16,col=cl,cex=.55);abline(h=c(-3,3),lty=3,col=P["ink"]);key(c("Posterior","Frontal","Other global"),c(P["accent"],P["ink"],P["neutral"]),16,pos="topright")
  vals<-c(183,2,1,4);panel("b","Fixed-region eligibility","","N2 episodes",c(.4,4.6),c(0,205),xaxis=FALSE);rect(1:4-.28,0,1:4+.28,vals,col=c(P["accent"],rep(P["neutral"],3)),border=NA);axis(1,at=1:4,labels=c("Included","Global","Posterior","Frontal"),tick=FALSE,cex.axis=.75);text(1:4,vals+7,vals,cex=.85)
  for(k in 1:2){d<-human[human$contrast==c("experience_presence","content_recall_given_experience")[k],];d$lab<-paste(c(primary_common_average="Primary",as_stored_reference="Stored ref.",unsaturated_common_average="Unsaturated",high_frequency_adjusted_common_average="High-frequency")[d$panel],ifelse(d$kind=="held_person_incremental_prediction","prediction","matched"));
    panel(letters[k+2],if(k==1)"Experience tests" else "Recall tests","Holm-adjusted p","",c(0,1.04),c(.5,nrow(d)+.5),yaxis=FALSE);axis(2,at=rev(seq_len(nrow(d))),labels=d$lab,tick=FALSE,las=1,cex.axis=.7);zero(v=.05);points(d$p_holm,rev(seq_len(nrow(d))),pch=ifelse(d$panel=="primary_common_average",16,1),col=ifelse(d$panel=="primary_common_average",P["accent"],P["ink"]),cex=.8)}
}

s7 <- function() {
  reset(2,2,mar=c(5.2,6.4,2.6,.9))
  prior<-read("prior/population_bridge/prediction_cases.csv");z<-prior[prior$account_class=="published_reference"&prior$variant=="pristine",];d<-aggregate(later_state_standardized_mse~participant_id+predictor,z,mean)
  labels<-c(matched_population_voltage_rate18="Population",matched_cortical_delay18="Cortical delays",original_scalar_map="Scalar",training_ar="AR",frozen_neural_transition="Neural transition",persistence="Persistence")
  selected<-names(labels);d<-d[d$predictor%in%selected,]
  panel("a","Original held-person forecasts","","Person-mean future-state MSE",c(.5,6.5),range_safe(d$later_state_standardized_mse,TRUE),xaxis=FALSE);axis(1,at=1:6,labels=labels,las=2,tick=FALSE,cex.axis=.7)
  for(person in unique(d$participant_id)){r<-d[d$participant_id==person,];r<-r[match(selected,r$predictor),];lines(1:6,r$later_state_standardized_mse,col=fade(P["ink"],.25));points(1:6,r$later_state_standardized_mse,pch=16,col=c(P["accent"],rep(P["ink"],5)),cex=.65)}
  compare<-read("human_bridge_comparisons.csv",TRUE);keep<-compare$comparator%in%selected;co<-compare[keep,];inds<-match(co$comparator,selected);co<-co[order(inds),];person_diff<-list()
  aa<-d[d$predictor=="matched_population_voltage_rate18",c("participant_id","later_state_standardized_mse")]
  panel("b","Paired forecast contrasts","Population minus comparator MSE","",c(-2.1,.3),c(.5,nrow(co)+.5),yaxis=FALSE);axis(2,at=rev(seq_len(nrow(co))),labels=labels[co$comparator],tick=FALSE,cex.axis=.75);zero(v=0)
  for(i in seq_len(nrow(co))){bb<-d[d$predictor==co$comparator[i],c("participant_id","later_state_standardized_mse")];m<-merge(aa,bb,by="participant_id");v<-m$later_state_standardized_mse.x-m$later_state_standardized_mse.y;points(v,rep(nrow(co)+1-i,length(v))+seq(-.09,.09,length.out=length(v)),pch=1,col=P["ink"],cex=.7);points(mean(v),nrow(co)+1-i,pch=16,col=P["accent"],cex=.9)}
  b<-balance[balance$scope=="all_incoming",];ratio<-b$maximum_absolute_sum_error_V/b$fixed_rounding_tolerance_V
  panel("c","Source-equation reconstruction","log10 error / rounding allowance","Fraction of population rows",c(-6,1),c(0,1));v<-log10(ratio[is.finite(ratio)&ratio>0]);lines(sort(v),seq_along(v)/length(v),col=P["accent"],lwd=1.5);zero(v=0)
  z<-balance[balance$scope=="all_incoming",];modes<-c("zero_input_tonic_and_fluctuation","zero_input_centered_fluctuation","paired_finite_pulse_response");counts<-sapply(modes,function(m)sum(is.finite(z$cancellation_for_inference[z$mode==m])));total<-sapply(modes,function(m)sum(z$mode==m));panel("d","Cancellation qualification","","Population rows",c(.5,3.5),c(0,max(total)*1.18),xaxis=FALSE);rect(1:3-.27,0,1:3+.27,total,col=P["neutral"],border=NA);rect(1:3-.27,0,1:3+.27,counts,col=P["accent"],border=NA);axis(1,at=1:3,labels=c("Tonic","Fluctuation","Pulse"),tick=FALSE);text(1:3,total+max(total)*.06,paste0(counts,"/",total),cex=.77)
}

figures <- list(Figure_1=fig1,Figure_2=fig2,Figure_3=fig3,Figure_4=fig4,Figure_5=fig5,
               Figure_S1=s1,Figure_S2=s2,Figure_S3=s3,Figure_S4=s4,Figure_S5=s5,Figure_S6=s6,Figure_S7=s7)
for(name in names(figures)) {
  height<-if(grepl("_S",name))160 else 155
  grDevices::cairo_pdf(file.path(out,paste0(name,".pdf")),width=185/25.4,height=height/25.4,family="Arial",pointsize=10.5,bg=CANVAS)
  render_figure(figures[[name]]);dev.off()
  grDevices::svg(file.path(out,paste0(name,".svg")),width=185/25.4,height=height/25.4,family="Arial",pointsize=10.5,bg=CANVAS)
  render_figure(figures[[name]]);dev.off()
  # Rasterize the canonical R PDF so PNG typography and clipping match it exactly.
  poppler<-Sys.which("pdftoppm");stopifnot(nzchar(poppler))
  status<-system2(poppler,c("-f","1","-singlefile","-r","400","-png",shQuote(file.path(out,paste0(name,".pdf"))),shQuote(file.path(out,name))))
  stopifnot(status==0)
  write.csv(do.call(rbind,lapply(HEADERS,function(h)data.frame(letter=h$letter,title=h$title,x_letter=h$x[1],x_title=h$x[2],y=h$y,plot_left=h$left,plot_right=h$right,plot_bottom=h$bottom,plot_top=h$top))),file.path(root,"provenance",paste0(name,"_headers.csv")),row.names=FALSE)
  cat(name,"exported\n")
}
for(bundle in c("Main_Figures","Supplementary_Figures")) {
  chosen<-if(bundle=="Main_Figures") names(figures)[1:5] else names(figures)[6:12]
  height<-if(bundle=="Main_Figures")155 else 160
  cairo_pdf(file.path(out,paste0(bundle,".pdf")),width=185/25.4,height=height/25.4,family="Arial",pointsize=10.5,bg=CANVAS)
  for(name in chosen)render_figure(figures[[name]]);dev.off()
}
capture.output(sessionInfo(),file=file.path(root,"provenance/R_sessionInfo.txt"))
write.csv(data.frame(file=list.files(out),bytes=file.info(file.path(out,list.files(out)))$size),file.path(root,"provenance/export_inventory.csv"),row.names=FALSE)
cat("Completed: 5 main figures, 7 supplementary figures; PDF, SVG, 400 dpi PNG.\n")
