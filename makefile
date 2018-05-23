# base url for auxiliary data download
URL = http://download.hygeos.com/POLYMER/auxdata
WGET = @wget -c -P

# mark these make targets as non-file targets, so they'll always trigger when checked
.PHONY: main all rebuild clean test auxdata_all auxdata_common auxdata_common auxdata_meris auxdata_olci auxdata_modisa auxdata_seawifs auxdata_viirs auxdata_msi

main:
	python setup.py build_ext --inplace

all: auxdata_all main ancillary

rebuild: clean all

clean:
	rm -rfv build
	rm -fv polymer/*.so

test: all
	nosetests test/test.py

ancillary:
	mkdir -p ANCILLARY/METEO/


auxdata_all: auxdata_common auxdata_meris auxdata_olci auxdata_modisa auxdata_seawifs auxdata_viirs auxdata_msi

auxdata_common: directories auxdata/generic/LUT.hdf auxdata/common/no2_climatology.hdf auxdata/common/trop_f_no2_200m.hdf auxdata/common/morel_fq.dat auxdata/common/AboveRrs_gCoef_w0.dat auxdata/common/AboveRrs_gCoef_w10.dat auxdata/common/AboveRrs_gCoef_w5.dat auxdata/common/aph_bricaud_1995.txt auxdata/common/aph_bricaud_1998.txt auxdata/common/morel_buiteveld_bsw.txt auxdata/common/palmer74.dat auxdata/common/pope97.dat auxdata/common/raman_westberry13.txt auxdata/common/astarmin_average_2015_SLSTR.txt auxdata/common/astarmin_average.txt auxdata/common/Matsuoka11_aphy_Table1_JGR.csv
	@echo "b88aadd272734634b756922ad5b6f439  auxdata/common/no2_climatology.hdf"            |md5sum -c -
	@echo "10350ad3441c9e76346f6429985f3c71  auxdata/common/trop_f_no2_200m.hdf"            |md5sum -c -
	@echo "7f3ba3b9ff13b9f135c53256d02a8b1b  auxdata/common/morel_fq.dat"                   |md5sum -c -
	@echo "44d9d702654ed7a35ba0a481a66be604  auxdata/common/AboveRrs_gCoef_w0.dat"          |md5sum -c -
	@echo "5f1eea393b1fda6d25f54ad34f93c450  auxdata/common/AboveRrs_gCoef_w10.dat"         |md5sum -c -
	@echo "f86841556820841ed0d623a69cbc9984  auxdata/common/AboveRrs_gCoef_w5.dat"          |md5sum -c -
	@echo "6ae4d62a28140e7221ad615ef4a59e8f  auxdata/common/aph_bricaud_1995.txt"           |md5sum -c -
	@echo "c998374a93b993123d6f85e9f627787d  auxdata/common/aph_bricaud_1998.txt"           |md5sum -c -
	@echo "7f178e809c8b8d4f379a26df7d258640  auxdata/common/morel_buiteveld_bsw.txt"        |md5sum -c -
	@echo "a7896ee2b35e09cacbeb96be69883026  auxdata/common/palmer74.dat"                   |md5sum -c -
	@echo "ba868100590c3248e14892c32b18955d  auxdata/common/pope97.dat"                     |md5sum -c -
	@echo "0dda3b7d9e2062abbb24f55f86ededf5  auxdata/common/raman_westberry13.txt"          |md5sum -c -
	@echo "c340ec49f1ad3214a4ee84a19652b7ac  auxdata/common/astarmin_average_2015_SLSTR.txt"|md5sum -c -
	@echo "56cd52dfaf2dab55b67398ac9adcbded  auxdata/common/astarmin_average.txt"           |md5sum -c -
	@echo "862c49b5dd19c9b09e451891ef11ce50  auxdata/common/Matsuoka11_aphy_Table1_JGR.csv" |md5sum -c -
	@echo "4cfc8b2ab76b1b2b2ea85611940ae6e2  auxdata/generic/LUT.hdf"                       |md5sum -c -
directories:
	@mkdir -p auxdata/common/
	@mkdir -p auxdata/generic/
auxdata/generic/LUT.hdf:
	$(WGET) auxdata/generic/ $(URL)/generic/LUT.hdf
auxdata/common/no2_climatology.hdf:
	$(WGET) auxdata/common/ $(URL)/common/no2_climatology.hdf
auxdata/common/trop_f_no2_200m.hdf:
	$(WGET) auxdata/common/ $(URL)/common/trop_f_no2_200m.hdf
auxdata/common/morel_fq.dat:
	$(WGET) auxdata/common/ $(URL)/common/morel_fq.dat
auxdata/common/AboveRrs_gCoef_w0.dat:
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w0.dat
auxdata/common/AboveRrs_gCoef_w10.dat:
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w10.dat
auxdata/common/AboveRrs_gCoef_w5.dat:
	$(WGET) auxdata/common/ $(URL)/common/AboveRrs_gCoef_w5.dat
auxdata/common/aph_bricaud_1995.txt:
	$(WGET) auxdata/common/ $(URL)/common/aph_bricaud_1995.txt
auxdata/common/aph_bricaud_1998.txt:
	$(WGET) auxdata/common/ $(URL)/common/aph_bricaud_1998.txt
auxdata/common/morel_buiteveld_bsw.txt:
	$(WGET) auxdata/common/ $(URL)/common/morel_buiteveld_bsw.txt
auxdata/common/palmer74.dat:
	$(WGET) auxdata/common/ $(URL)/common/palmer74.dat
auxdata/common/pope97.dat:
	$(WGET) auxdata/common/ $(URL)/common/pope97.dat
auxdata/common/raman_westberry13.txt:
	$(WGET) auxdata/common/ $(URL)/common/raman_westberry13.txt
auxdata/common/astarmin_average_2015_SLSTR.txt:
	$(WGET) auxdata/common/ $(URL)/common/astarmin_average_2015_SLSTR.txt
auxdata/common/astarmin_average.txt:
	$(WGET) auxdata/common/ $(URL)/common/astarmin_average.txt
auxdata/common/Matsuoka11_aphy_Table1_JGR.csv:
	$(WGET) auxdata/common/ $(URL)/common/Matsuoka11_aphy_Table1_JGR.csv


auxdata_meris: auxdata/meris/smile/v2/sun_spectral_flux_rr.txt auxdata/meris/smile/v2/central_wavelen_rr.txt auxdata/meris/smile/v2/sun_spectral_flux_fr.txt auxdata/meris/smile/v2/central_wavelen_fr.txt
	@mkdir -p auxdata/meris
	@echo "477aef2509b692b599dc0c4db3134b94  auxdata/meris/smile/v2/sun_spectral_flux_rr.txt" |md5sum -c -
	@echo "249b2cf1934f2e42fe5133b7fe739cce  auxdata/meris/smile/v2/central_wavelen_rr.txt"   |md5sum -c -
	@echo "e807d872e16513c8ee40f68a4b57d784  auxdata/meris/smile/v2/sun_spectral_flux_fr.txt" |md5sum -c -
	@echo "08b210bf3c4fe4c78d4db0c068820579  auxdata/meris/smile/v2/central_wavelen_fr.txt"   |md5sum -c -
auxdata/meris/smile/v2/sun_spectral_flux_rr.txt:
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/sun_spectral_flux_rr.txt
auxdata/meris/smile/v2/central_wavelen_rr.txt:
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/central_wavelen_rr.txt
auxdata/meris/smile/v2/sun_spectral_flux_fr.txt:
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/sun_spectral_flux_fr.txt
auxdata/meris/smile/v2/central_wavelen_fr.txt:
	$(WGET) auxdata/meris/smile/v2/ $(URL)/meris/smile/v2/central_wavelen_fr.txt


auxdata_modisa: auxdata/modisa/HMODISA_RSRs.txt
	@mkdir -p auxdata/modisa
	@echo "2868db4dd5cb9c5782ddc574168e0e29  auxdata/modisa/HMODISA_RSRs.txt" |md5sum -c -
auxdata/modisa/HMODISA_RSRs.txt:
	$(WGET) auxdata/modisa/ $(URL)/modisa/HMODISA_RSRs.txt


auxdata_seawifs: auxdata/seawifs/SeaWiFS_RSRs.txt
	@mkdir -p auxdata/seawifs
	@echo "a58950c5f1b9be06f862fe1938723ea1  auxdata/seawifs/SeaWiFS_RSRs.txt" |md5sum -c -
auxdata/seawifs/SeaWiFS_RSRs.txt:
	$(WGET) auxdata/seawifs/ $(URL)/seawifs/SeaWiFS_RSRs.txt


auxdata_viirs: auxdata/viirs/VIIRSN_IDPSv3_RSRs.txt
	@mkdir -p auxdata/viirs
	@echo "4479d74b44c4423cf8ae192abad6bad2  auxdata/viirs/VIIRSN_IDPSv3_RSRs.txt" |md5sum -c -
auxdata/viirs/VIIRSN_IDPSv3_RSRs.txt:
	$(WGET) auxdata/viirs/ $(URL)/viirs/VIIRSN_IDPSv3_RSRs.txt

auxdata_msi: auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv
	@mkdir -p auxdata/msi
	@echo "1f815b74a94246ab99f607894c9483ec  auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv" |md5sum -c -
	@echo "414d614f2d15125498c6d7517c4e2f76  auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv" |md5sum -c -
auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv:
	$(WGET) auxdata/msi/ $(URL)/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv
auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv:
	$(WGET) auxdata/msi/ $(URL)/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
