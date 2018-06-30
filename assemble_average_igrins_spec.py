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
	re-written 2018 to be orders
	
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

def align(wls_shifted,line_observed,line_reference,f):
	#align(flux_shifted,uncs_shifted, wls_shifted,line_observed,line_reference):
	#this is where we apply a wavelength shift
	#the shift value is a change in wavelength, and from the line
	
	#actual equation for correcting Doppler shifts
	c=299792.458
	rv=c*((line_observed/line_reference)-1.)
	print 'The measured rv is ', rv
	f.write('The measured rv is '+np.str(rv)+' (km/s) \n')
	wls_rest=wls_shifted/(1.+(rv/c))		
			
	#so by doing it this way, i am assuming the real "value" of the pixels here is the
	#shifted value
	#flux_fixed=np.interp(wls_shifted, wls_rest,flux_shifted)
	#uncs_fixed=np.interp(wls_shifted, wls_rest,uncs_shifted)
	
	#return [flux_fixed,uncs_fixed]
	return [rv,wls_rest]

def gaussian(x,amp,cen,wid):
	#just defining the gaussian to fit the sky line
	return amp*np.exp(-(x-cen)**2/wid)
	
def find_line_center(flux, wls, cutoffs, line='abs'):
	#this is where we fit a gaussian to a line and find the center in pixel space
	blueend,redend=cutoffs


	#using just the specified region of the spectra
	region=scipy.where((blueend < wls) & (wls < redend))
	fluxes_out=flux[region]
	wls_out=wls[region]
	
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
	init_vals=[np.max(f), np.mean(wls_out), 10.]
	best_vals, covar = curve_fit(gaussian, wls_out, f, p0=init_vals)
	center = best_vals[1]
	print best_vals
	print 'The center of the line is at ', center

	#plot to ensure that you are actually measuring the line center	
	plotnow='yes'
	if plotnow == 'yes':
		#approx center is
		plt.axvline(x=center, color="blue")
		plt.plot(wls_out, f, color="magenta")
		plt.text(center,-0.01,"wave ~"+np.str(center))
		plt.title('Finding center of given line')
		plt.show()
		plt.close()

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


def band_spectra(obsdir, specfile, band, f, f2):
	wls=[]
	Is=[]
	sns=[]
	
	#step 1: read in the data	
	specpath=obsdir+specfile	
	spec = pyfits.getdata(specpath)
	wlsol = pyfits.getdata(specpath,1)
	snr = pyfits.getdata(specpath,5)	
	dataheader = pyfits.getheader(specpath)

	print '*** File ***'
	f.write('*** File ***'+'\n')
	
	f.write('Individual spectrum: '+specfile+' \n')
	print 'Individual spectrum: ', specfile
	#this is assuming you used refine_igrins_telluric_corr.py
	#if you did not, then re-write this line to get the snr from the .sn.fits file
	
 
	
	#step 2: learn about the data
	object=dataheader['OBJECT',0]
	date=dataheader['UTDATE',0]
	amstart=dataheader['AMSTART',0]
	amend=dataheader['AMEND',0]
	am=0.5*(np.float(amstart)+np.float(amend))
	tel=dataheader.get('TELESCOP')
	exptime=dataheader.get('EXPTIME')
	
	f.write('OBJECT: '+ object+'\n')
	f.write('DATE: '+ date+'\n')
	f.write('am dets:'+ np.str(amstart)+'\t'+np.str(amend)+'\n')
	f.write('average am: '+ np.str(am)+'\n')
	f.write('telescope: '+ tel+'\n')
	f.write('exposure time: '+ np.str(exptime)+'\n')

	print 'OBJECT: ', object
	print 'DATE: ', date
	print 'am dets:', np.str(amstart)+'\t'+np.str(amend)
	print 'average am: ', am
	print 'telescope: ', tel
	print 'exposure time: ', exptime
	
	if band == 'K':
		f2.write(specfile+"\t")
		f2.write(np.str(object)+'\t'+np.str(date)+'\t'+np.str(tel)+'\t'+np.str(exptime)+'\t'+np.str(am)+'\t')

	target_id=specfile.split(".J")
	target_id=target_id[0].split("_")
	target_id=target_id[1]+"_"+target_id[2]

	#step 3. check and fix the size of the spectra (number of orders)
		
	wlsol,spec,snr=fix_num_orders(wlsol,spec,snr)

	time=0
	#step 4: start stepping through order by order.
	for wl, I, sn in zip(wlsol, spec,snr):

		#step 4.a. convert units
		wl=wl*1.e4 #from 1.4 to 1.4e4 as expected (um to Ang)

		#step 4.b. quality check
		#lets chop the bad data a bit. Greg did a cut off of 8 to -6.
		med=np.nanmean(I)
		bad=scipy.where(-25.*med > I)
		bad2=scipy.where(I > 25.*med)
		
		if med > 0.:
			I[bad]=np.nan
			sn[bad]=np.nan
			I[bad2]=np.nan
			sn[bad2]=np.nan		
	
		#step 4.c. lets normalize the spectra (or find the value to later anyhow)
		#lets find an average continuum value, as this will vary for each spectrum, and normalize
		#have to find the correct order first, then we will just keep that value
		if band == 'K':
			if wl[0] < 21920:
				if wl[0] > 21600:
					#this should be the correct order for the continuum spot!
					cont_start=scipy.where(wl > 21920)
					cont_start=cont_start[0]
					print 'order for estimating continuum:', wl[0]
					cont_region=cont_start[0:800]
					Icont=I[cont_region]
	
					#first cut off the top % for being crazy
					top=np.nanpercentile(Icont, 95)
					no_top=scipy.where(Icont < top)
					Icont=Icont[no_top]
					#now define the part to keep
					keep=np.nanpercentile(Icont, 80)

					where_keep=scipy.where(Icont > keep)
					aIcont=Icont[where_keep]
					Icont=np.nanmedian(aIcont)

	
					#plot to ensure that you are actually estimating the continuum
					plotnow='no'
					if plotnow == 'yes':
						plt.plot(wl[cont_region], I[cont_region], color="red")
						plt.plot(wl[cont_region][no_top][where_keep], aIcont, color="blue")
						plt.axhline(y=Icont, color='black')
						plt.axhline(y=np.nanmedian(I[cont_region]), color='green')
						plt.title('Finding the continuum')
						plt.show()
						plt.close()

		if band == 'H':
			if wl[0] < 16000:#15500:
				if wl[0] > 15800:#15400:
					
					#this should be the correct order for the continuum spot!
					cont_start=scipy.where(wl > 15950)#15500)
					cont_start=cont_start[0]
					print 'order for estimating continuum:', wl[0]
					cont_region=cont_start[0:800]#1000]
					Icont=I[cont_region]
	
					#first cut off the top % for being crazy
					top=np.nanpercentile(Icont, 95)
					no_top=scipy.where(Icont < top)
					Icont=Icont[no_top]
					#now define the part to keep
					keep=np.nanpercentile(Icont, 80)

					where_keep=scipy.where(Icont > keep)
					aIcont=Icont[where_keep]
					Icont=np.nanmedian(aIcont)

	
					#plot to ensure that you are actually estimating the continuum
					plotnow='no'
					if plotnow == 'yes':
						plt.plot(wl[cont_region], I[cont_region], color="red")
						plt.plot(wl[cont_region][no_top][where_keep], aIcont, color="blue")
						plt.axhline(y=Icont, color='black')
						plt.axhline(y=np.nanmedian(I[cont_region]), color='green')
						plt.title('Finding the continuum')
						plt.show()
						plt.close()

		#step 4.d. Find the shift that you will need by measuring the center of a line		
		#have to find the order where your line is
		if band == 'K':
			#assuming the first order you reach (as they go in descending wavelength)
			#will work.
			if wl[0] < aligning_region[0]:
				if time == 0:
					print 'Aligning in order: ', wl[0]
					foundcenter=find_line_center(I, wl, aligning_region)
					#now determine the shift
					time = 1
				
		if band == 'H':
			#need to use the shift found from before			
			foundcenter=0.
		
		wls.append(wl)
		Is.append(I)
		sns.append(sn)
	outdata=[wls, Is, sns, Icont, foundcenter, dataheader, target_id]	
	return	outdata	
				

"""
This is the body of the code.
"""

### INPUT ###
configfile=sys.argv[1]
cfg=read_config(configfile)
obsdir=cfg["obsdir"]
obsdir_out=cfg["obsdir_out"]
filein=obsdir+cfg["filein"]
filename_out=cfg["filename_out"]	
aligning_region=[np.float(cfg["linestart"]),np.float(cfg["linestop"])]
if "lab_center" in cfg:
	reference = 'lab'
	lab_center=np.float(cfg["lab_center"])	
else:
	reference = "first"

### BEGIN PROGRAM ###	
#I also want to save a list of the input files that it actually uses. Yes we read them
#in, its just a good cross check.
#this will be one per spectra
foundcenters=[]
save_spec_files=[]
ids=[]

filename_out_txt=obsdir_out+filename_out+".txt" #text file out on info
f=open(filename_out_txt, 'w')
f.write('Combining spectra! \n')

filename_out_tbl=obsdir_out+filename_out+".tbl" #short table
f2=open(filename_out_tbl, 'w')
f2.write('Combining spectra! \n')


#step through all of the spectra
specfiles=np.loadtxt(filein, skiprows=0, dtype="str")
if specfiles.size == 1:
	specfiles=[specfiles.tolist()]

#INPUT ONLY K, and it will do H (if you like)
bands=['K','H']

for band in bands:
	### STEP A: Get all the individual spectra ready to combine
	print 'Combining band ', band
	f.write('Combining band = '+band+'\n')	

	#do it for one band first, all the spectra and then combine
	#this is where we will start saving our data
	#all of these will be order by order
	saved_wavesol=[]
	saved_fluxes_normed=[]
	saved_snrs=[]

	for i,specfile in enumerate(specfiles):
		if band == 'K':
			specfilename=specfile.split("DCH")
			specfile=specfilename[0]+"DCK"+specfilename[1]
			
		save_spec_files.append(specfile)
		if i == 0:
			if band == 'K':
				f2.write('File \t Object \t UT Date \t Telescope \t Exp Time \t Airmass \t RV \t Standard \t Sequence \t c2d Name'+'\n')

		wl, I, sn, Icont, foundcenter, dataheader,id= band_spectra(obsdir, specfile, band, f,f2)
		ids.append(id)
		
		#apply the normalizations!
		normed=I/Icont
		snrs=sn
		
		### am going to need to cycle through again?
		if band == 'K':
			foundcenters.append(foundcenter)
			f.write('Measured line center'+np.str(foundcenter)+'\n')	

		elif band == 'H':
			foundcenter=foundcenters[i]
			print 'Using the shift from K'
			
		### STEP B: Align and normalize the spectra
		#no longer order by order, can do this to the entire thing
	
		#step 1.a. Apply any shifts to align the spectra
		if reference == 'first':
			#find the first spectrum
			line_reference=foundcenters[0]
		elif reference == 'lab':
			line_reference = lab_center
		
		f.write('Finding rv shift from line at '+np.str(line_reference)+'\n')	

		if foundcenter != line_reference:
			rv,wl=align(wl,foundcenter,line_reference,f)
				
		#step 1.b. Get the spectra on the same wavelength solution (from first spectra)
		if i == 0:
			wavesol=wl
		else:
			for i in range(len(wavesol)):
				normed[i]=np.interp(wavesol[i],wl[i],normed[i])
				snrs[i]=np.interp(wavesol[i],wl[i],snrs[i])
		
		#and we will call all of the fluxes normed now
		saved_fluxes_normed.append(normed)
		saved_snrs.append(snrs)
		
		#save some stuff to a table
		if band == 'K':
			f2.write(np.str(rv)+"\n")
		

		
	#now work to combine all of the spectra in one band
	### STEP C: Take the weighted mean
	print 'Now taking the weighed average of ', len(saved_fluxes_normed)
	f.write('***    \n')	
	f.write('Now taking the weighed average of '+np.str(len(saved_fluxes_normed))+' spectra \n')	

	#now lets find the mean! well, the weighted mean. so we need to do a lot with the uncs!
	#weight by the unc (well, 1/sigma^2). so lets assume that snr=1/unc. so weights are snr^2?
	saved_snrs=np.array(saved_snrs)
	w=(saved_snrs)**2
	###so I will be weighting by the uncertainties. 


	#giving 0 weights where there is a nan, wherever that may be! Yes, i know i did it to myself.
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

	sfilename_out=filename_out+'_'+band+'.fits'
	f.write('*** File out *** \n')
	f.write('The combined spectra is being saved to '+sfilename_out+'\n')

	sfilename_out=obsdir_out+sfilename_out
	#write the primary data, and fill out the header
	hdu=pyfits.PrimaryHDU()
	hdu.writeto(sfilename_out, clobber=True)
	#copy the target header
	header=dataheader
	header["EXTNAME"] = 'SPEC_COMBINED'
	header["N_SPECS"]=(len(saved_fluxes_normed),"Number of spec_div_a0v stars that have been combined")
	header.add_comment("This science spectrum was created by taking a weighted average")
	header.add_comment("Flux files: "+np.str(save_spec_files))
	pyfits.update(sfilename_out,averaged_flux, header)


	#add the rest of the extensions
	header["EXTNAME"]="WAVELENGTH"
	pyfits.append(sfilename_out,wavesol, header)
	header["EXTNAME"]="UNC_FLUX"
	pyfits.append(sfilename_out,unc_ave_units, header)
	header["EXTNAME"]="SNR"
	pyfits.append(sfilename_out,snr_ave, header)

	print '*** File out ***'
	print 'The combined spectra is being saved to ', sfilename_out
	
	### STEP E: Lets plot!

	plot='yes'
	if plot == 'yes':
		#overlay them on top of each other to check
		fig, (ax1,ax2)=plt.subplots(2,sharex=False)


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

		#plot 1: each spectra spaced apart and labeled (zoomed in a bit)
		#plot 2: all on top of each other, with the average output in black 
	
		n=0
		if band == 'H':
			textspot=17000
		else:
			textspot=21700
		for each_flux, snrs,id,linecolor in zip(saved_fluxes_normed,saved_snrs,ids,colors):
			ax1.plot(wavesol,each_flux+0.15*n,color=linecolor,linestyle = 'None', marker=".")
			ax1.text(textspot,1+0.15*n,id+" snr="+'%.2f' % np.nanmedian(snrs))
			ax2.plot(wavesol,each_flux,color=linecolor,linestyle = 'None', marker=".")
			n=n+1
		for i in range(len(wavesol)):
			#error bar won't work unless i go order by order
			ax2.errorbar(wavesol[i],averaged_flux[i],yerr=unc_ave_units[i], color='k')
		
		#now that you have gone through all of the spectra, show the plots
		plt.xlabel('Wavelength $\AA$')			
		plt.ylabel('Flux')
		ax1.set_title('Final Combined Spectrum')
		ax1.set_ylim([0.7,1.1+0.15*len(saved_fluxes_normed)])
		ax2.set_ylim([0,2.])
		if band == 'K':
			ax1.set_xlim([21000,22500])
			ax2.set_xlim([19000,25000])
		elif band == 'H':
			ax1.set_xlim([16500,17500])
			ax2.set_xlim([14500,18300])
		
		print 'The combined snr is', '%.2f' % np.nanmedian(snr_ave)
		f.write('The combined snr is '+'%.2f' % np.nanmedian(snr_ave)+'\n')	
		plt.show()	
		plt.close()
		
	f.write('\n')	
	f.write('\n')	
	f.write('\n')	
f.close()
f2.close()
