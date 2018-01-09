import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy
import sys
import numpy as np
from matplotlib.pyplot import cm 
from scipy.optimize import curve_fit

"""
This program stitch_single_igrins_spec.py is to read in a telluric corrected spectra 
and stitch together the orders. 

author: 
	Kim Sokal 2017
	
input:
	config file that points to all the data. 
	it has the following entries:
		obsdir [example: = /Users/IGRINS_reduced_data/TWHya/ ]
		specfile	[telluric corrected spectrum. Assumed to be in obsdir. example: = SDCK_20150127_0152.spec_a0v.fits]
		filename_out [example: = twhya_stitched.fits]

output:
	single, order-combined spectrum			
	
how to run:	
	python stitch_single_igrins_spec.py stitch_single_igrins_spec.cfg
	
	* while it runs you will see a final plot popup to check the final output
	
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

### BEGIN PROGRAM ###

configfile=sys.argv[1]
cfg=read_config(configfile)
specfile=cfg["specfile"]
obsdir=cfg["obsdir"]
filename_out=cfg["filename_out"]	
	
#step 1: read in the data	
specpath=obsdir+specfile
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

#step 3: lets get all the data so we can combine it
### STITCH ALL THE ORDERS TOGETHER
wls,fluxes,sns= stitch(wlsol, spec, snr, cutoverlap='yes')	
	
#step 4: save to a fits file
#write the primary data, and fill out the header
hdu=pyfits.PrimaryHDU()
hdu.writeto(filename_out, clobber=True)
#copy the target header
header=dataheader
header["EXTNAME"] = 'SPEC_STITCHED'
header.add_comment("This spectrum was created by combining the orders")
pyfits.update(filename_out,fluxes, header)

#add the rest of the extensions
header["EXTNAME"]="WAVELENGTH"
pyfits.append(filename_out,wls, header)
###This one is new!
header["EXTNAME"]="SNR"
pyfits.append(filename_out,sns, header)

print '*** File out ***'
print 'The divided spectra is being saved to ', filename_out

#step 5: plot
uncs=sns**-1
uncs_units=uncs*fluxes
plt.errorbar(wls,fluxes, yerr=uncs_units, marker='s')
plt.xlabel('Wavelength $\AA$')			
plt.ylabel('Flux')
plt.ylim([-1.e6,1.e6])
plt.xlim([19000,25000])
plt.show()
			