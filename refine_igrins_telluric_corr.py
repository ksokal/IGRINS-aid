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
				* note that the program will pull out the date itself 
				  (as plp data are stored as /outdata/20150127/)
				  you only need the directory above that level
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
	f=flipped-np.nanmin(flipped)
	
	
	#now fit it with a gaussian to find the center!
	n=len(f)
	newpix=np.arange(n)

	init_vals=[np.max(fluxes_out), np.mean(newpix), np.mean(newpix)/0.1]
	best_vals, covar = curve_fit(gaussian, newpix, f, p0=init_vals)
	center = best_vals[1]

	#plot to ensure that you are actually measuring the line center	
	plotnow='yes'
	if plotnow == 'yes':
		plt.plot([center]*n, f, color="blue")
		plt.plot(newpix, f, color="magenta")
		plt.title('Finding center of sky line')
		plt.show()
		plt.close()
	
	center_value=center+np.min(pixelsin[blueend])
	return center


"""
This is the body of the code.
"""

### BEGIN PROGRAM ###

configfile=sys.argv[1]
cfg=read_config(configfile)
specfile_target=cfg["specfile_target"]
specfile_standard=cfg["specfile_standard"]
obsdir=cfg["obsdir"]
filename_out=cfg["filename_out"]	
band=cfg["band"]
	
#step 1.a: read in the observed data

find_obsdate_target=specfile_target.split("_")
obsdate_target=find_obsdate_target[1]
find_snr_target=specfile_target.split(".spec.")
snrfile_target=find_snr_target[0]+".sn."+find_snr_target[1]
vegafile=find_snr_target[0]+".spec_a0v."+find_snr_target[1]

specpath_target=obsdir+obsdate_target+'/'+specfile_target
snrpath_target=obsdir+obsdate_target+'/'+snrfile_target
vegapath_target=obsdir+obsdate_target+'/'+vegafile

spec_target = pyfits.getdata(specpath_target)
wlsol_target = pyfits.getdata(specpath_target,1)
snr_target = pyfits.getdata(snrpath_target)
vega = pyfits.getdata(vegapath_target,4)

filename_out=find_snr_target[0]+filename_out+find_snr_target[1]
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
print 'average am: ', am_target

#step 2.a: read in the standard data

find_obsdate_standard=specfile_standard.split("_")
obsdate_standard=find_obsdate_standard[1]
find_snr_standard=specfile_standard.split(".spec.")
snrfile_standard=find_snr_standard[0]+".sn."+find_snr_standard[1]

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

#step 3: need to find any pixel shift between the spectra, by measuring a sky line
#unfortunately that means that I go through the entire spectra 

save_centers=[]
num=0

for spec,wlsol,snr in zip([spec_target,spec_standard],[wlsol_target,wlsol_standard],[snr_target,snr_standard]):

	fluxes=[]
	wls=[]
	sns=[]
	
	for wl, I, sn in zip(wlsol, spec,snr):
		
		#convert units
		wl=wl*1.e4 #from 1.4 to 1.4e4 as expected (um to Ang i think)
		
		#can have just positive numbers here. won't matter later, only looking for line center
		good=scipy.where(I < 0.0)		
		
		#combine all the data	   		
		wls.extend(wl)
		fluxes.extend(I)
		sns.extend(sn)

	#lets sort these by wavelength, since the orders are random and have overlap
	#### ACTUALLY I CANT DO THIS _ remember it messes some up. need to fix.
	sorts=np.argsort(wls)
	wls=np.array(wls)
	fluxes=np.array(fluxes)
	wls=wls[sorts]
	fluxes=fluxes[sorts]
	sns=np.array(sns)
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
			region=[21740,21749] #if it has problems, try editting a bit
			#this is just a line that i found that works well. you can play with it, and just run again!

		blueend=scipy.where(wls > region[0])
		findblue=pix[blueend]
		blueend_pix=np.min(findblue)
	
		redend=scipy.where(wls < region[1])
		findred=pix[redend]
		redend_pix=np.max(findred)
	
	if len(wls) != keep:		
		print '\t The spectra have a different number of orders!'

	#ok lets find the shift!
	foundcenter=find_line_center(pix, fluxes, wls,[blueend_pix,redend_pix])
	save_centers.append(foundcenter)

	#yep, all that work to find the center of that line
	num=+1

shift = save_centers[1]-save_centers[0]	
print '*** shift ***'
print 'The target and standard are shifted by {0} pixels'.format(shift)

#step 4: order by order, apply the shift and divide the spectra
spec_target_a0v=[]
wlsol=[]
snr=[]

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
	div=spec_target[order]/spec_standard[order]*vega[order]
	spec_target_a0v.append(div)
	wlsol_order=wlsol_standard[order]
	wlsol.append(wlsol_order)
	#adding in quadrature
	unc_order=(snr_standard[order]**-2+snr_target[order]**-2)**0.5
	snr.append(unc_order**-1)
	
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


#step 6: plot them to check	
### this is the slow part

#overlay them on top of each other to check
f, (ax1,ax2)=plt.subplots(2,sharex=True)

#Plot the input target and standard spectra
ax1.plot(wlsol_target,snr_standard,color="magenta",linestyle = 'None', marker=".")
ax1.plot(wlsol_target,snr_target,color="green",linestyle = 'None', marker=".")
ax1.plot([],[], color="magenta", label="Standard")
ax1.plot([],[], color="green", label="Target")

#Plot your output telluric corrected spectra		
ax2.plot(wlsol,spec_target_a0v,color="black",linestyle = 'None', marker=".")
ax2.set_ylim([-1.e7,1.e7])

plt.xlabel('Wavelength $\AA$')			
plt.ylabel('Flux')
ax1.set_title('Final Telluric Corrected Spectra') 
ax1.legend()   		
plt.show()	
plt.close()