# base url for auxiliary data download
URL = http://download.hygeos.com/POLYMER/auxdata
WGET = @wget -q -c -P

all: auxdata_all polymer

polymer:
	python setup.py build_ext --inplace

rebuild: clean all

clean:
	rm -fv polymer/*.html
	rm -fv polymer/*.pyc
	rm -fv polymer/*.c
	rm -fv polymer/*.so

test: all
	nosetests -v

auxdata_all: auxdata_common auxdata_meris auxdata_olci auxdata_modisa auxdata_seawifs auxdata_viirs auxdata_msi

auxdata_common:
	@mkdir -p auxdata/common/
	$(WGET) auxdata/common/ $(URL)/common/no2_climatology.hdf
	$(WGET) auxdata/common/ $(URL)/common/trop_f_no2_200m.hdf
	$(WGET) auxdata/common/ $(URL)/common/morel_fq.dat
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w0.dat
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w10.dat
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w5.dat
	$(WGET) auxdata/common/ $(URL)/common/aph_bricaud_1995.txt
	$(WGET) auxdata/common/ $(URL)/common/morel_buiteveld_bsw.txt
	$(WGET) auxdata/common/ $(URL)/common/palmer74.dat
	$(WGET) auxdata/common/ $(URL)/common/pope97.dat
	$(WGET) auxdata/common/ $(URL)/common/raman_westberry13.txt
	$(WGET) auxdata/common/ $(URL)/common/astarmin_average_2015_SLSTR.txt
	$(WGET) auxdata/common/ $(URL)/common/astarmin_average.txt
	@echo b88aadd272734634b756922ad5b6f439 auxdata/common/no2_climatology.hdf            |md5sum -c -
	@echo 10350ad3441c9e76346f6429985f3c71 auxdata/common/trop_f_no2_200m.hdf            |md5sum -c -
	@echo 7f3ba3b9ff13b9f135c53256d02a8b1b auxdata/common/morel_fq.dat                   |md5sum -c -
	@echo 44d9d702654ed7a35ba0a481a66be604 auxdata/common/AboveRrs_gCoef_w0.dat          |md5sum -c -
	@echo 5f1eea393b1fda6d25f54ad34f93c450 auxdata/common/AboveRrs_gCoef_w10.dat         |md5sum -c -
	@echo f86841556820841ed0d623a69cbc9984 auxdata/common/AboveRrs_gCoef_w5.dat          |md5sum -c -
	@echo 6ae4d62a28140e7221ad615ef4a59e8f auxdata/common/aph_bricaud_1995.txt           |md5sum -c -
	@echo 7f178e809c8b8d4f379a26df7d258640 auxdata/common/morel_buiteveld_bsw.txt        |md5sum -c -
	@echo a7896ee2b35e09cacbeb96be69883026 auxdata/common/palmer74.dat                   |md5sum -c -
	@echo ba868100590c3248e14892c32b18955d auxdata/common/pope97.dat                     |md5sum -c -
	@echo 0dda3b7d9e2062abbb24f55f86ededf5 auxdata/common/raman_westberry13.txt          |md5sum -c -
	@echo c340ec49f1ad3214a4ee84a19652b7ac auxdata/common/astarmin_average_2015_SLSTR.txt|md5sum -c -
	@echo 56cd52dfaf2dab55b67398ac9adcbded auxdata/common/astarmin_average.txt           |md5sum -c -

auxdata_meris:
	@mkdir -p auxdata/meris
	$(WGET) auxdata/meris/          $(URL)/meris/LUTB.hdf
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/sun_spectral_flux_rr.txt
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/central_wavelen_rr.txt
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/sun_spectral_flux_fr.txt
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/central_wavelen_fr.txt
	@echo 68f7ee2d82ac2b8de85af511e6398460  auxdata/meris/LUTB.hdf |md5sum -c -
	@echo 477aef2509b692b599dc0c4db3134b94  auxdata/meris/smile/v2/sun_spectral_flux_rr.txt |md5sum -c -
	@echo 249b2cf1934f2e42fe5133b7fe739cce  auxdata/meris/smile/v2/central_wavelen_rr.txt   |md5sum -c -
	@echo e807d872e16513c8ee40f68a4b57d784  auxdata/meris/smile/v2/sun_spectral_flux_fr.txt |md5sum -c -
	@echo 08b210bf3c4fe4c78d4db0c068820579  auxdata/meris/smile/v2/central_wavelen_fr.txt   |md5sum -c -

auxdata_olci:
	@mkdir -p auxdata/olci
	$(WGET) auxdata/olci/ $(URL)/olci/LUT.hdf
	@echo 535ab472aca939352c14ff8e9d11eae2  auxdata/olci/LUT.hdf |md5sum -c -

auxdata_modisa:
	@mkdir -p auxdata/modisa
	$(WGET) auxdata/modisa/ $(URL)/modisa/LUTB.hdf
	@echo 6e098f2ee54daba73147dc93259c23d6  auxdata/modisa/LUTB.hdf |md5sum -c -

auxdata_seawifs:
	@mkdir -p auxdata/seawifs
	$(WGET) auxdata/seawifs/ $(URL)/seawifs/LUT.hdf
	@echo 0af6cba18e7db320a11c05ca7f106906  auxdata/seawifs/LUT.hdf |md5sum -c -

auxdata_viirs:
	@mkdir -p auxdata/viirs
	$(WGET) auxdata/viirs/ $(URL)/viirs/LUT.hdf
	@echo a1232d073512add31c4a0c0e40eb97ba  auxdata/viirs/LUT.hdf |md5sum -c -

auxdata_msi:
	@mkdir -p auxdata/msi
	$(WGET) auxdata/msi/ $(URL)/msi/LUT.hdf
	@echo 2d24fb7f0107518c59544bdb220a6e9d  auxdata/msi/LUT.hdf |md5sum -c -

