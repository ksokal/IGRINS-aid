import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy
import sys
import numpy as np
from matplotlib.pyplot import cm 
from scipy.optimize import curve_fit

"""
This program refine_igrins_telluric_corr.py is to take some of the IGRINS plp products
to refine the telluric correction. It will divide a science spectra by a standard spectra,
in the same fashion as the plp, with the following intended differences:
- don't need to run from the pipeline (just a quick program)
- will find the pixel center of a given sky line and then shift the standard spectra in pixel space
before dividing the target by this (thus possible to use standards from different nights).

author: 
	Kim Sokal 2017

input:
	config file with the target and standard spectra names (which are from the plp)
	it has the following entries:
		specfile_target [example:  ='SDCK_20150127_0152.spec.fits']
		specfile_standard	[example: ='SDCK_20150127_0156.spec.fits']
		obsdir	[example: ='/Users/observations/IG_plp_reduced_data/outdata/']
				* note that the program expects plp format. It will pull out the date itself 
				  (as plp data are stored as /outdata/20150127/)
				  you only need the directory above that level (i.e. outdata/)
		filename_out	[this will be added before .fits in the target filename. example: ='.spec_a0v.']
		band ['H' or 'K'. example: = 'K']
	
output: 
	telluric corrected target spectra
		specfile_target+filename_out.fits

how to run:
	python refine_igrins_telluric_corr.py refine_igrins_telluric_corr.cfg
	
	* while it runs you will see 3 plots pop up. The first two are showing the fit to 
	the sky lines - if the vertical line doesn't match the center then something is wrong.
	The last is to check the final output.

"""
def read_config(configfile):
	cfg=dict()
	f=open(configfile)
	for row in f:
		if not row.strip() or row.lstrip().startswith('#'):
			continue
		option, value = [r.strip() for r in row.split('=')]
		cfg[option]=value
	
	return cfg
	
def align_by_f_interp(shiftvalue,flux_shifted,uncs_shifted,wvsol):
	#this is where we apply the pixel shift
	
	length=len(flux_shifted)
	pixs=np.arange(length)
	shifted_pixs=pixs+shiftvalue
	
	#so by doing it this way, i am assuming the real "value" of the pixels here is the
	#shifted value
	flux_fixed=np.interp(shifted_pixs, pixs,flux_shifted)
	uncs_fixed=np.interp(shifted_pixs, pixs,uncs_shifted)
		
	return [flux_fixed,uncs_fixed,wvsol]

def gaussian(x,amp,cen,wid):
	#just defining the gaussian to fit the sky line
	return amp*np.exp(-(x-cen)**2/wid)
	
def find_line_center(pixelsin, normed, wls, cutoffs):
	#this is where we fit a gaussian to a sky line and find the center in pixel space
	blueend_pix,redend_pix=cutoffs
	
	#using just the specified region of the spectra
	blueend=scipy.where(pixelsin > blueend_pix)
	fluxes_out=normed[blueend]
	pixels=pixelsin[blueend]
	wls=wls[blueend]
	
	redend=scipy.where(pixels < redend_pix)
	fluxes_out=fluxes_out[redend]
	pixels=pixels[redend]
	wls=wls[redend]
	
	#trying to make it look like a normal gaussian.
	#meaning positive, as the sky lines are in absorption
	#shouldn't effect where the center is too much
	fluxes_out=np.array(fluxes_out)
	flipped=1./fluxes_out
	f=flipped-np.nanmedian(flipped)
	
	
	#now fit it with a gaussian to find the center!
	n=len(f)
	newpix=np.arange(n)

	init_vals=[np.max(fluxes_out), np.mean(newpix), np.mean(newpix)/3]
	best_vals, covar = curve_fit(gaussian, newpix, f, p0=init_vals)
	center = best_vals[1]

	#plot to ensure that you are actually measuring the line center	
	plotnow='yes'
	if plotnow == 'yes':
		plt.plot(pixelsin, normed, color="green")
		plt.plot([center+np.min(pixelsin[blueend])]*n, fluxes_out, color="blue")
		#plt.plot(newpix+np.min(pixelsin[blueend]), f, color="magenta")
		#plt.plot([center]*n, f, color="blue")
		#plt.plot(newpix, f, color="magenta")
		plt.title('Finding center of sky line')
		plt.xlim(cutoffs[0]-50, cutoffs[1]+50)
		plt.show()
		plt.close()

	center_value=center+np.min(pixelsin[blueend])
	return center


def fix_num_orders(wls, fluxes,sns):
	keep = 26#53248 # 26 orders times 2048 pixels per order
	diff=keep-len(wls)
	#print 'there were', len(wls)
	if diff >0:
		#add in nans for any missing data. it is probably for the early part!
		startwave=np.nanmin(wls)
		#print 'start', startwave
		if startwave < 1000:
			#wrong units for some reason
			startwave=startwave*1.e4
		if startwave > 18600:
			#one order
			add=np.array([0.]*2048)
			wls=np.insert(wls,-1,add, axis=0)
			fluxes=np.insert(fluxes,-1,add, axis=0)
			sns=np.insert(sns,-1,add, axis=0)
			if startwave > 18800:
				#two orders
				wls=np.insert(wls,-1,add, axis=0)
				fluxes=np.insert(fluxes,-1,add, axis=0)
				sns=np.insert(sns,-1,add, axis=0)
				if startwave > 19000:
					#three orders
					wls=np.insert(wls,-1,add, axis=0)
					normed=np.insert(normed,-1,add, axis=0)
					sns=np.insert(sns,-1,add, axis=0)
		if len(wls) != keep:
			diff=keep-len(wls)
			add=np.array([0.]*diff)
			wls=np.insert(wls,-1,add, axis=0)
			fluxes=np.insert(fluxes,-1,add, axis=0)
			sns=np.insert(sns,-1,add, axis=0)
	#print 'there are now', len(wls)		
	return [wls,fluxes,sns]

"""
This is the body of the code.
"""

### BEGIN PROGRAM ###

configfile=sys.argv[1]
cfg=read_config(configfile)
file_target=cfg["specfile_target"]
file_standard=cfg["specfile_standard"]
obsdir=cfg["obsdir"]
base_filename_out=cfg["filename_out"]	
obsdir_out=cfg["obsdir_out"]	
band=cfg["band"]
if "bvc" in cfg:
	bary_correct='True'
	bvc= np.float(cfg["bvc"])
else:
	bary_correct='False'
if "recipedir" in cfg:	
	recipe_info ='yes'
	recipedir=cfg["recipedir"]
else:
	recipe_info ='no'	
	
#step 1.a: read in the observed data

targetfile=file_target.split(".fits")
find_obsdate_target=targetfile[0].split("_")
obsdate_target=find_obsdate_target[1]
filenumber_target=find_obsdate_target[2]

specfile_target=targetfile[0]+".spec.fits"
snrfile_target=targetfile[0]+".sn.fits"
vegafile=targetfile[0]+".spec_a0v.fits"

specpath_target=obsdir+obsdate_target+'/'+specfile_target
snrpath_target=obsdir+obsdate_target+'/'+snrfile_target
vegapath_target=obsdir+obsdate_target+'/'+vegafile

spec_target = pyfits.getdata(specpath_target)
wlsol_target = pyfits.getdata(specpath_target,1)
snr_target = pyfits.getdata(snrpath_target)
vega = pyfits.getdata(vegapath_target,4) 

filename_out=obsdir_out+targetfile[0]+"."+base_filename_out+".spec_a0v.fits" #spectra
filename_out_txt=obsdir_out+targetfile[0]+"."+base_filename_out+".spec_a0v.txt" #text file out on info
f=open(filename_out_txt, 'w')
f.write('Performing a telluric correction \n')
print 'Performing a telluric correction'
print specfile_target

dataheader_target = pyfits.getheader(specpath_target)

#step 1.b: learn about the observed data
object_target=dataheader_target['OBJECT',0]
date_target=dataheader_target['UTDATE',0]
amstart=dataheader_target['AMSTART',0]
amend=dataheader_target['AMEND',0]
am_target=0.5*(np.float(amstart)+np.float(amend))
print '*** Target ***'
print 'OBJECT: ', object_target
print 'DATE: ', date_target
print 'am dets:', np.str(amstart)+'\t'+np.str(amend)
print 'average am: ', am_target
f.write('*** Target ***'+'\n')
f.write('SPEC FILE: '+ specfile_target+'\n')
f.write('OBJECT: '+ object_target+'\n')
f.write('DATE: '+ date_target+'\n')
f.write('am dets:'+ np.str(amstart)+'\t'+np.str(amend)+'\n')
f.write('average am: '+ np.str(am_target)+'\n')

tel=dataheader_target.get('TELESCOP')
exptime=dataheader_target.get('EXPTIME')
acqtime=dataheader_target.get('ACQTIME') #get local date
obs=dataheader_target.get('OBSERVER')

print '**********************'
print 'Target observing info from header:'
print 'Object \t UT Date \t Telescope \t Exp Time \t Airmass \t Observers'
print np.str(object_target)+'\t'+np.str(date_target)+'\t'+np.str(tel)+'\t'+np.str(exptime)+'\t'+np.str(am_target)+'\t'+np.str(obs)
f.write('**********************'+'\n')
f.write('Target observing info from header:'+'\n')
f.write('Object \t UT Date \t Telescope \t Exp Time \t Airmass \t Observers'+'\n')
f.write(np.str(object_target)+'\t'+np.str(date_target)+'\t'+np.str(tel)+'\t'+np.str(exptime)+'\t'+np.str(am_target)+'\t'+np.str(obs)+'\n')

if recipe_info =='yes':
	#step 1.c. how many frames are in there?
	#in dir '.../recipe_logs' instead of outdata. then date.recipes
	#recipedir=/Volumes/IGRINS_reduced/reduced_data/recipe_logs/
	recipelog=recipedir+obsdate_target+'.recipes'
	#each line will be 'observed name', 'target type', group1 = file # for some, group 2, exptime, recipe, obsids, frametypes
	filenumber_target=np.int(filenumber_target)
	f_recipe=open(recipelog, 'r')
	
	for line in f_recipe.readlines():
		split=line.split(",")
		test_sky=split[5]
		test_sky=test_sky.strip(" ")
		if test_sky != 'SKY':
			find_file=split[6].split()
			if find_file[0] == np.str(filenumber_target):
				print 'recipe file:'
				print line
				f.write('recipe file: \n')
				f.write(line+'\n')

	
#step 1.d. make sure the order numbers are the same
a,vega,b=fix_num_orders(wlsol_target,vega,snr_target)
wlsol_target,spec_target,snr_target=fix_num_orders(wlsol_target,spec_target,snr_target)


#step 2.a: read in the standard data

standardfile=file_standard.split(".fits")
find_obsdate_standard=standardfile[0].split("_")
obsdate_standard=find_obsdate_standard[1]

specfile_standard=standardfile[0]+".spec.fits"
snrfile_standard=standardfile[0]+".sn.fits"

specpath_standard=obsdir+obsdate_standard+'/'+specfile_standard
snrpath_standard=obsdir+obsdate_standard+'/'+snrfile_standard

spec_standard = pyfits.getdata(specpath_standard)
wlsol_standard = pyfits.getdata(specpath_standard,1)
snr_standard = pyfits.getdata(snrpath_standard)

dataheader_standard = pyfits.getheader(specpath_standard)

#step 2.b: learn about the standard data
object_standard=dataheader_standard['OBJECT',0]
date_standard=dataheader_standard['UTDATE',0]
amstart=dataheader_standard['AMSTART',0]
amend=dataheader_standard['AMEND',0]
am_standard=0.5*(np.float(amstart)+np.float(amend))
print '*** standard ***'
print 'OBJECT: ', object_standard
print 'DATE: ', date_standard
print 'average am: ', am_standard

f.write('*** standard ***'+'\n')
f.write('SPEC FILE: '+ specfile_standard+'\n')
f.write('OBJECT: '+ object_standard+'\n')
f.write('DATE: '+ date_standard+'\n')
f.write('average am: '+ np.str(am_standard)+'\n')
f.write('start am: '+np.str(amstart)+'\n')
f.write('end am: '+np.str(amend)+'\n')

#step 2.d: correct for # of orders
wlsol_standard,spec_standard,snr_standard=fix_num_orders(wlsol_standard,spec_standard,snr_standard)

#step 3: need to find any pixel shift between the target and standard spectra, by measuring a sky line.
#(there is a pixel shift in general, esp. when we move between telescopes). Order of < or a couple of pixels.
#unfortunately that means that I have to go through the entire spectra 

save_centers=[]
num=0

for spec,wlsol,snr in zip([spec_target,spec_standard],[wlsol_target,wlsol_standard],[snr_target,snr_standard]):

	fluxes=[]
	wls=[]
	sns=[]
	
	for wl, I, sn in zip(wlsol, spec,snr):
		
		#convert units
		wl=wl*1.e4 #from 1.4 to 1.4e4 as expected (um to Ang i think)
		
		#combine all the data	   		
		wls.extend(wl)
		fluxes.extend(I)
		sns.extend(sn)

	#lets sort these by wavelength, since the orders are random and have overlap
	sorts=np.argsort(wls)
	wls=np.array(wls)
	fluxes=np.array(fluxes)
	sns=np.array(sns)
	wls=wls[sorts]
	fluxes=fluxes[sorts]
	sns=sns[sorts]
	
	#lets make sure our 2 spectra are the same length - the pipeline currently produced different
	#numbers of orders depending on the flat fields. so we need to fix this.
	
	pix=np.arange(len(fluxes))
	#first one first
	if num==0:
		keep=len(fluxes)
		
		#find the cut offs for the line we will be using
		#choosing some region of the spectra
		if band == 'K':
			region=[21740,21752] #if it has problems, try editing a bit. also [22145,22165]?
			#this is just a line that i found that works well. you can play with it, and just run again!
		elif band == 'H':
			region=[16452,16458] #if it has problems, try [16452,16458]?16429,16431]
		blueend=scipy.where(wls > region[0])
		findblue=pix[blueend]
		blueend_pix=np.min(findblue)
	
		redend=scipy.where(wls < region[1])
		findred=pix[redend]
		redend_pix=np.max(findred)
	
	#ok lets find the shift!
	foundcenter=find_line_center(pix, fluxes, wls,[blueend_pix,redend_pix])
	save_centers.append(foundcenter)

	#yep, all that work to find the center of that line
	num=+1

shift = save_centers[1]-save_centers[0]	
print '*** shift ***'
print 'The target and standard are shifted by {0} pixels'.format(shift)

f.write('*** shift ***'+'\n')
f.write('The target and standard are shifted by {0} pixels'.format(shift)+'\n')

#step 4: order by order, apply the shift and divide the spectra
spec_target_a0v=[]
wlsol=[]
snr=[]

if bary_correct == 'True':
		print 'correcting for a barycenter velocity of ', bvc
		f.write('correcting for a barycenter velocity of '+np.str(bvc)+'\n')
		
for order in range(len(spec)):
	#this assumes that the spectra have the same number of orders and are in the same order

	##ok, so lets shift the standard spectra here. by the amount in pixels.
	pix_order=np.arange(len(wlsol_standard))
	#it will interpolate
	spec_standard[order],snr_standard[order],wlsol_standard[order]=align_by_f_interp(shift,spec_standard[order],snr_standard[order],wlsol_standard[order])	
		
	### when dividing: thinking about if we need to normalize them in any way. but
	## that means just multiplying by some factor - so the answer is no then,
	# it is not mathematically necessary.
		
	#fyi - keeping the target wavesol so we can just use that vega to get the correct shape
	
	### why does this work for vega if its a diff wavesol? not a huge problem bc it is mostly featureless. 
	div=spec_target[order]/spec_standard[order]*vega[order]
	
	#do we want to correct for the barycenter velocity? set at the beginning by adding to the cfg file
	if bary_correct == 'True':
		#from Greg
		#BVC= (correct in superlog) km/s
		bvc=np.float(bvc)
		c=299792.458
		#corrected wave = ((orig wave)*(1+BVC/c))
		wlsol_target[order]=wlsol_target[order]*(1.+bvc/c)
	
	spec_target_a0v.append(div)
	wlsol_order=wlsol_target[order]
	#wlsol_order=wlsol_standard[order] used to use this
	wlsol.append(wlsol_order)
	#adding in quadrature
	unc_order=(snr_standard[order]**-2+snr_target[order]**-2)**0.5
	snr.append(unc_order**-1)
	

#final output is spec_target_a0v and wlsol_target
#wlsol_target should be the same for each telluric


#need to normalize it
spec_target_a0v=np.array(spec_target_a0v)

wlsol=np.array(wlsol)*1.e4
if band == 'K':
	cont_region=scipy.where(wlsol > 21920)
elif band == 'H':
	cont_region=scipy.where(wlsol > 15885)
cont_region=cont_region[0:800]

Icont=spec_target_a0v[cont_region]
Icont=np.nanmedian(Icont)
spec_target_a0v=spec_target_a0v/Icont
	
#step 5: save to a new fits file
#the extensions are a bit like the plp output
#except now with uncs (that incorporate the a0v division as well)

#write the primary data, and fill out the header
hdu=pyfits.PrimaryHDU()
hdu.writeto(filename_out, clobber=True)
#copy the target header
header=dataheader_target
header["EXTNAME"] = 'SPEC_DIVIDE_A0V'
header["STD"]=(specfile_standard,"Standard spectra used")
header.add_comment("This spectrum was created by dividing by the standard spectra and multiplying by a generic Vega (as in the plp)")
header.add_comment("The standard pixels were shifted by an offset of "+np.str(shift) )
if bary_correct == 'True':
	header.add_comment("Barycenter velocity corrected, by a bvc= "+np.str(bvc) )
pyfits.update(filename_out,spec_target_a0v, header)


#add the rest of the extensions
header["EXTNAME"]="WAVELENGTH"
pyfits.append(filename_out,wlsol_target, header)
header["EXTNAME"]="TGT_SPEC"
pyfits.append(filename_out,spec_target, header)
header["EXTNAME"]="A0V_SPEC"
pyfits.append(filename_out,spec_standard, header)
header["EXTNAME"]="VEGA_SPEC"
pyfits.append(filename_out,vega, header)
###This one is new!
header["EXTNAME"]="SNR"
pyfits.append(filename_out,snr, header)

print '*** File out ***'
print 'The divided spectra is being saved to ', filename_out

f.write('*** File out ***'+'\n')
f.write('The divided spectra is being saved to '+filename_out)
f.close()

#step 6: plot them to check	
### this is the slow part

#overlay them on top of each other to check
#f, (ax1,ax2)=plt.subplots(2,sharex=True)
f, (ax1,ax2,ax3,ax4)=plt.subplots(4)#,sharex=True)
wlsol_target=wlsol

if band == 'K':
	zoomout=[19000,25000]
	zoomin=[21500,22000]
elif band == 'H':
	zoomout=[14500,18000]
	zoomin=[16200,16500]

#Plot the input target and standard spectra as is
ax1.plot(wlsol_target,spec_standard,color="magenta",linestyle = 'None', marker=".")
ax1.plot(wlsol_target,spec_target,color="green",linestyle = 'None', marker=".")
ax1.plot([],[], color="magenta", label="Standard")
ax1.plot([],[], color="green", label="Target")
ax1.set_ylim([-10.,np.nanmedian(spec_standard)*5.])
ax1.set_xlim(zoomout)


#Plot the output telluric corrected spectra: zoomed out	
ax2.plot(wlsol_target,spec_target_a0v,color="black",linestyle = 'None', marker=".")
ax2.set_ylim([0.5, 2.])
ax2.set_xlim(zoomout)

#Plot the output telluric corrected spectra: zoomed in	
ax3.plot(wlsol_target,spec_target_a0v,color="black",linestyle = 'None', marker=".")
ax3.set_ylim([0.5, 2.])
ax3.set_xlim(zoomin)

#Plot the input target and standard spectra, zoomed in
ax4.plot(wlsol_target,spec_standard,color="magenta",linestyle = 'None', marker=".")
ax4.plot(wlsol_target,spec_target,color="green",linestyle = 'None', marker=".")
ax4.plot([],[], color="magenta", label="Standard")
ax4.plot([],[], color="green", label="Target")
ax4.set_ylim([-10.,np.nanmedian(spec_standard)*5.])
ax4.set_xlim(zoomin)

plt.xlabel('Wavelength $\AA$')			
plt.ylabel('Flux')
ax1.set_title('Telluric Correcting Spectra') 
ax1.legend()   		

figname=obsdir_out+targetfile[0]+"."+base_filename_out+'.jpg'
plt.savefig(figname)

plt.show()	
plt.close()



