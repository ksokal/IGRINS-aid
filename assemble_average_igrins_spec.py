import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy
import sys
import numpy as np
from matplotlib.pyplot import cm 
from scipy.optimize import curve_fit

"""
This program assemble_average_igrins_spec.py is to read in telluric corrected spectra 
and combine them. It first finds any potential wavelength shift - although one must first input a
reasonable stellar line to do this with, this will vary for every target.
Then it aligns the spectra and then takes an weighted average of all of the spectra.

author: 
	Kim Sokal 2017
	
input:
	config file that points to all the data. 
	it has the following entries:
		obsdir [example: = /Users/IGRINS_reduced_data/TWHya/ ]
		filein	[A list of all the spectra. Assumed to be in obsdir. example: = all_obs.txt]
		filename_out [example: = w_ave_twhya.fits]
		band ['H' or 'K'. example: = 'K']
		line_region [Wavelength range around a single line for aligning. example: = []]

output:
	weighted average spectrum			
	
how to run:	
	python assemble_average_igrins_spec.py  assemble_average_igrins_spec.cfg
	
	* while it runs you will see several plots pop up: 
		- for each spectrum, you will see the gaussian fit to the given line.
		 if the vertical line doesn't match the center,	then something is wrong 
		 (try a different line or play with the input region if the spectra are good).
		- the last is to check the final output
	
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

def align_by_f_interp(shiftvalue,flux_shifted,uncs_shifted, wls_shifted,cutoffs=None):
	#this is where we apply the pixel shift

	length=len(flux_shifted)
	pixs=np.arange(length)

	#for some collections of spectra, the shift might be large.
	#fix for the change in velocity resolution across the whole spectrum
	change=(1.-1./45000.)*np.array(wls_shifted)
	change=np.array(change)-np.array(wls_shifted)#just the  diff across the spec
	ave=np.average(cutoffs)
	change=change-change[ave.astype(int)+1]
	shifted_pixs=pixs+shiftvalue-change			
	
	#so by doing it this way, i am assuming the real "value" of the pixels here is the
	#shifted value
	flux_fixed=np.interp(shifted_pixs, pixs,flux_shifted)
	uncs_fixed=np.interp(shifted_pixs, pixs,uncs_shifted)

	return [flux_fixed,uncs_fixed]

def gaussian(x,amp,cen,wid):
	#just defining the gaussian to fit the sky line
	return amp*np.exp(-(x-cen)**2/wid)
	
def find_line_center(pixelsin, normed, wls, cutoffs, line='abs'):
	#this is where we fit a gaussian to a line and find the center in pixel space
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
	#meaning positive, as some lines are in absorption
	#shouldn't effect where the center is too much
	fluxes_out=np.array(fluxes_out)
	
	#decide here if it is emission or absorption!
	line='abs'
	
	if line == 'abs':
		flipped=1./fluxes_out
	elif line == 'em':
		flipped=fluxes_out
	f=flipped-np.nanmedian(flipped)
	
	#now fit it with a gaussian to find the center!
	n=len(f)
	newpix=np.arange(n)

	init_vals=[np.max(fluxes_out), np.mean(newpix), np.mean(newpix)/0.1]
	best_vals, covar = curve_fit(gaussian, newpix, f, p0=init_vals)
	center = best_vals[1]

	#plot to ensure that you are actually measuring the line center	
	plotnow='yes'
	if plotnow == 'yes':
		#approx center is
		approx_center=np.round(center)
		approx_center=approx_center.astype(int)
		plt.plot([center]*n, f, color="blue")
		plt.plot(newpix, f, color="magenta")
		plt.text(newpix[approx_center],-0.01,"wave ~"+np.str(wls[approx_center]))
		plt.title('Finding center of given line')
		plt.show()
		plt.close()

	center_value=center+np.min(pixelsin[blueend])
	return center

def stitch(wlsol, spec, snr, cutoverlap='yes'):
	#this will combine all of the orders of your spectra
	fluxes=[]
	wls=[]
	sns=[]
	
	tracker=0
	all_wls_starts=[]

	#it goes order by order
	for wl, I, sn in zip(wlsol, spec,snr):
		
		#convert units
		wl=wl*1.e4 #from 1.4 to 1.4e4 as expected (um to Ang)
		#just to find correct ordering of the orders, keep the starting wavelength
		all_wls_starts.append(wl[0])

		#lets chop the bad data a bit. Greg did a cut off of 8 to -6.
		med=np.nanmean(I)
		bad=scipy.where(-25.*med > I)
		bad2=scipy.where(I > 25.*med)
		
		I[bad]=np.nan
		sn[bad]=np.nan
		I[bad2]=np.nan
		sn[bad2]=np.nan		
				
		#Would you like to chop off the overlap regions? You really should.
		if cutoverlap=='yes':
			if tracker==0:
				#this is the first order it read in. keep for later.
				#it will be a redder order
				red_wl=wl
				red_I=I
				red_sn=sn
				tracker=tracker+1
			
			else:
				
				bluest=np.nanmin(red_wl)				
				reddest=np.nanmax(wl)
				
				overlap=scipy.where(bluest < wl)
				red_overlap=scipy.where(red_wl < reddest)
				
				if len(red_I[red_overlap]) > 10:
					# lets interpolate
					#want to keep the blue wavelength sol rather than the red
					blue_wv=wl[overlap]
					red_wv=red_wl[red_overlap]

					#also will want the flux and sn
					blue_sn=sn[overlap]
					blue_I=I[overlap]
					
					### Working in the overlap region
					
					# 1. is there any data? if it is all nan, then need to keep 
					#the other one.
					
					isnotnan_blue=np.isfinite(blue_I)
					isnotnan_red=np.isfinite(red_I[red_overlap])
					if np.sum(isnotnan_blue) !=  0:
						#making sure they are not all nans
						
						overlap_flux=blue_I
						overlap_sn=blue_sn	
									
						#interpolating the flux to the same wavelength sol			
						red_flux_fixed=np.interp(blue_wv, red_wv,red_I[red_overlap])
						red_sn_fixed=np.interp(blue_wv, red_wv,red_sn[red_overlap])
					
						#so at this point we have all on the blue_wv solution. 
										
						# 2. now i need to figure out where the red is diff from the blue by
						#more than a certain percentage, then we will keep part
					
						#need to check if the red isn't just all nan
						if np.sum(isnotnan_red) !=  0:
							diff=(blue_I-red_flux_fixed)/red_flux_fixed
										
							length=len(overlap_flux)
							half=np.rint(length/2.)
							half=half.astype(int)
												
							#only on the red HALF, make the comparison
							huge_diff=scipy.where(np.abs(diff[half:]) > 0.1)
							
							huge_diff=huge_diff+half
							huge_diff=huge_diff.astype(int)

							#So here we are now setting the overlap flux to the red value IF is meets
							#the criteria
							overlap_flux[huge_diff]=red_flux_fixed[huge_diff]					
							overlap_sn[huge_diff]=red_sn_fixed[huge_diff]					
					
							red_I[red_overlap]=np.nan 
							red_sn[red_overlap]=np.nan 
							red_wl[red_overlap]=np.nan
							I[overlap]=overlap_flux
							sn[overlap]=overlap_sn

									
				### THIS IS WHERE WE ACTUALLY SAVE. Whatever is called red_*
   		
				wls.append(red_wl)
				fluxes.append(red_I)
				sns.append(red_sn)
								
				#now cut the red part of the current order and make it the one that will
				#be red next

				tracker=tracker+1
				red_wl=wl
				red_I=I
				red_sn=sn				

   				if tracker == len(wlsol):
   					#the last order
					wls.append(red_wl)
					fluxes.append(red_I)
					sns.append(red_sn)
					
		else:
			print 'Overlap regions left un-altered'
			#combine all the data	   		
			wls.append(wl)
			fluxes.append(I)
			sns.append(sn)
	#so i original wanted to sort by wavelength but it is better instead to
	#put the orders in order.
	#we sort by the starting wavelength of the order
	sorts=np.argsort(all_wls_starts)
	wls=[wls[i] for i in sorts]
	wls=np.array(wls)
	wls=wls.flatten()
	fluxes=[fluxes[i] for i in sorts]
	fluxes=np.array(fluxes)
	fluxes=fluxes.flatten()
	sns=[sns[i] for i in sorts]
	sns=np.array(sns)
	sns=sns.flatten()

	return [wls,fluxes,sns]

"""
This is the body of the code.
"""

### INPUT ###

configfile=sys.argv[1]
cfg=read_config(configfile)
obsdir=cfg["obsdir"]
filein=obsdir+cfg["filein"]
filename_out=cfg["filename_out"]	
band=cfg["band"]
aligning_region=[np.float(cfg["linestart"]),np.float(cfg["linestop"])]

### BEGIN PROGRAM ###
	
#this is where we will start saving our data
saved_wavesol=[]
saved_fluxes_normed=[]
saved_snrs=[]

#I also want to save a list of the input files that it actually uses. Yes we read them
#in, its just a good cross check.
save_spec_files=[]
ids=[]

### STEP A: Get all the individual spectra ready to combine

num=0
#step through all of the spectra
specfiles=np.loadtxt(filein, skiprows=0, dtype="str")

for specfile in specfiles:

	#step 1: read in the data	
	specpath=obsdir+specfile
	save_spec_files.append(specfile) 
	spec = pyfits.getdata(specpath)
	wlsol = pyfits.getdata(specpath,1)
	snr = pyfits.getdata(specpath,5)	
	#this is assuming you used refine_igrins_telluric_corr.py
	#if you did not, then re-write this line to get the snr from the .sn.fits file
	
	dataheader = pyfits.getheader(specpath)

	#step 2: learn about the data
	object=dataheader['OBJECT',0]
	date=dataheader['UTDATE',0]
	amstart=dataheader['AMSTART',0]
	amend=dataheader['AMEND',0]
	am=0.5*(np.float(amstart)+np.float(amend))
	sourcedets=object+date
	
	print 'Spectra # ', num
	print 'OBJECT: ', object
	print 'DATE: ', date
	print 'average am: ', am

	
	#step 3: lets get all the data so we can combine the orders
	### stitch all the orders together
	wls,fluxes,sns= stitch(wlsol, spec, snr, cutoverlap='yes')	

	#step 4: lets normalize the spectra
	#lets find an average continuum value, as this will vary for each spectrum, and normalize
	if band == 'K':
		cont_start=scipy.where(wls > 21300)#21750)
	if band == 'H':
		cont_start=scipy.where(wls > 15500)
	cont_start=cont_start[0]
	cont_region=cont_start[0:1000]
	Icont=fluxes[cont_region]
	cont=np.nanmedian(Icont)
	
	normed=fluxes/cont
	#and we will call all of the fluxes normed now
		
	#step 5. check the size of the spectra
	
	#lets make sure our spectra are the same length
	keep = 53248 # 26 orders times 2048 pixels per order
	diff=keep-len(wls)
	
	pix1=np.arange(len(wls))	
	
	if diff >0:
		#add in nans for any missing data. it is probably for the early part!
		startwave=np.nanmin(wls)
		if startwave > 18600:
			#one order
			add=np.array([np.nan]*2048)
			wls=np.insert(wls,0,add)
			normed=np.insert(normed,0,add)
			sns=np.insert(sns,0,add)
			if startwave > 18800:
				#two orders
				wls=np.insert(wls,0,add)
				normed=np.insert(normed,0,add)
				sns=np.insert(sns,0,add)
				if startwave > 19000:
					#three orders
					wls=np.insert(wls,0,add)
					normed=np.insert(normed,0,add)
					sns=np.insert(sns,0,add)
		if len(wls) != keep:
			diff=keep-len(wls)
			add=np.array([np.nan]*diff)
			wls=np.insert(wls,-1,add)
			normed=np.insert(normed,-1,add)
			sns=np.insert(sns,-1,add)

	### STEP B: Align the spectra
	#now lets align the spectra to each other using some line
		
	pix=np.arange(len(wls))	

	### need to find any shift
	#first one first - this is your reference spectrum now 
	if num==0:
		#set this one as your final wavelength solution and the reference to align
		wavesol=wls

		#find where this falls in the spectra		
		reference_blueend=scipy.where(wls > aligning_region[0])
		findblue=pix[reference_blueend]
		reference_blueend_pix=np.min(findblue)
		#print 'the blue end is', np.min(findblue), 'at wave',  reference_blueend_pix

		reference_redend=scipy.where(wls < aligning_region[1])
		findred=pix[reference_redend]
		reference_redend_pix=np.max(findred)
		#print 'the red end is', np.max(findred), reference_redend_pix
	
		#lets set the reference pixel number
		foundcenter=find_line_center(pix, normed, wls,[reference_blueend_pix,reference_redend_pix])
		reference=foundcenter
		shift=0.
		
		#print 'wavelength at start of the reference spectrum', wls[0], np.nanmin(wls)
		
	if num != 0:	

		pix=np.arange(len(wls))

		blueend=scipy.where(wls > aligning_region[0])
		findblue=pix[blueend]
		blueend_pix_now=np.min(findblue)
		#print 'the blue end is', np.min(findblue), blueend_pix_now, 'at wave', wls[blueend_pix_now]
		#print 'compared to', reference_blueend_pix,  'at wave', wavesol[reference_blueend_pix]
		extra=blueend_pix_now-reference_blueend_pix
		#print 'the difference is', extra
		#instead of doing this again for the red, find the offset difference "extra"

		#ok lets find the shift!
		foundcenter=find_line_center(pix, normed, wls,[reference_blueend_pix+extra,reference_redend_pix+extra])
		shift=foundcenter-reference+extra
		print 'the pixel shift is', shift
			
		##ok, so lets shift the spectra here. by the amount in pixels.
		normed,sns=align_by_f_interp(shift, normed,sns, wls,cutoffs=[reference_blueend_pix+extra,reference_redend_pix+extra])

	#now we are saving the shifted spectra
	#the wavelength solution is from your reference spectrum (the first in the list)

	saved_fluxes_normed.append(normed)
	saved_snrs.append(sns)

	ids.append(sourcedets)
	num=num+1

### STEP C: Take the weighted mean

#now lets find the mean! well, the weighted mean. so we need to do a lot with the uncs!
#weight by the unc (well, 1/sigma^2). so lets assume that snr=1/unc. so weights are snr^2?
saved_snrs=np.array(saved_snrs)
w=(saved_snrs)**2
###so I will be weighting by the uncertainties. 


#giving 0 weights where there is a nan, wherever that may be!
where_nan=np.isnan(w)
where_nan=np.invert(np.isfinite(w))
w[where_nan]=0.0
saved_fluxes_normed=np.array(saved_fluxes_normed)
where_nan=np.isnan(saved_fluxes_normed)
saved_fluxes_normed[where_nan]=0.0
w[where_nan]=0.0

#the final maths to get the weighted average!
averaged_flux=(np.sum(w*saved_fluxes_normed, axis=0))/(np.sum(w, axis=0))


#then for determining the average uncertainty
#see https://www.colorado.edu/physics/phys2150/phys2150_sp14/phys2150_lec4.pdf
unc_ave=(np.sum(w, axis=0))**-0.5 ### note that this is the fractional uncertainty
snr_ave=unc_ave**-1
###also, probably want to save the real uncs, not the fractional unc
unc_ave_units=unc_ave*averaged_flux

#step D: save to a new fits file
#the order is a bit like the plp, except with the uncs incorporating the a0v division

#write the primary data, and fill out the header
hdu=pyfits.PrimaryHDU()
hdu.writeto(filename_out, clobber=True)
#copy the target header
header=dataheader
header["EXTNAME"] = 'SPEC_COMBINED'
header["N_SPECS"]=(len(saved_fluxes_normed),"Number of spec_div_a0v stars that have been combined")
header.add_comment("This science spectrum was created by taking a weighted average")
header.add_comment("Flux files: "+np.str(save_spec_files))
pyfits.update(filename_out,averaged_flux, header)


#add the rest of the extensions
header["EXTNAME"]="WAVELENGTH"
pyfits.append(filename_out,wavesol, header)
header["EXTNAME"]="UNC_FLUX"
pyfits.append(filename_out,unc_ave_units, header)
header["EXTNAME"]="SNR"
pyfits.append(filename_out,snr_ave, header)

print '*** File out ***'
print 'The combined spectra is being saved to ', filename_out

	
### STEP E: Lets plot!

plot='yes'
if plot == 'yes':
	fig = plt.figure()

	nums=np.arange(len(saved_fluxes_normed))

	#define some colors
	#i prefer to have the colors repeat than a continous set
	#for many spectra, the differences can be very subtle!
	
	if len(specfiles) > 6:
		colors=cm.rainbow(np.linspace(0,1,len(specfiles)/2))
		colors=colors.tolist()
		colors=colors*2 
	else:	
		colors=cm.rainbow(np.linspace(0,1,len(specfiles)))
		colors=colors.tolist()
	colors.append('magenta')

	
	for each_flux, snrs,id,linecolor in zip(saved_fluxes_normed,saved_snrs,ids,colors):
		plt.plot(wavesol,each_flux,color=linecolor,linestyle = 'None', marker=".")
	plt.errorbar(wavesol,averaged_flux,yerr=unc_ave_units, color='k')

	#now that you have gone through all of the spectra, show the plots
	plt.xlabel('Wavelength $\AA$')			
	plt.ylabel('Flux')
	plt.title('Final Combined Spectrum')
	plt.ylim([-10,10])
	plt.xlim([19000,25000])
	plt.show()	
	plt.close()

