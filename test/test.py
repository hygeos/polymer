import unittest
import os
from polymer.main import run_atm_corr
from polymer.level1_msi import Level1_MSI
from polymer.level1_meris import Level1_MERIS
from polymer.level1_olci import Level1_OLCI
from polymer.level1_netcdf import Level1_NETCDF
from polymer.level2_nc import Level2_NETCDF
from polymer.level2_hdf import Level2_HDF
from polymer.ancillary import Ancillary_NASA
from polymer.level2 import Level2
from numpy.testing import assert_allclose, assert_equal
import tempfile
import numpy as np

class Test_MERIS(unittest.TestCase):
    filename = '/mfs/user/francois/TESTCASES/MERIS/GlintMediteranea/MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1'

    @classmethod
    def setUpClass(self):
        pass

    def test1(self):
        run_atm_corr(
            Level1_MERIS(
                self.filename,
                sline=5774,     scol=299,
                eline=5774+100, ecol=299+150,
                ),
            Level2('memory'),
            )

    def test_MM01(self):
        run_atm_corr(
            Level1_MERIS(
                self.filename,
                sline=5774,     scol=299,
                eline=5774+100, ecol=299+150,
                ),
            Level2('memory'),
            water_model = 'MM01_FOQ',
            )

    def test_window(self):

        l2a = run_atm_corr(
                Level1_MERIS(
                    self.filename,
                    sline=5774,     scol=299,
                    eline=5774+100, ecol=299+150,
                    ),
                Level2('memory'),
                force_initialization=True,
                )

        l2b = run_atm_corr(
                Level1_MERIS(
                    self.filename,
                    sline=5774+50,     scol=299+50,
                    eline=5774+50+100, ecol=299+50+150,
                    blocksize=30,
                    ),
                Level2('memory'),
                force_initialization=True,
                )

        assert_allclose(
                l2a.Rw[50:60,50:60,:],
                l2b.Rw[:10,:10,:]
                )

    def test_output_netcdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, 'output.nc')
            run_atm_corr(
                Level1_MERIS(
                    self.filename,
                    sline=5774,     scol=299,
                    eline=5774+100, ecol=299+150,
                    ),
                Level2_NETCDF(target),
                multiprocessing=-1,
                )

    def test_output_hdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, 'output.hdf')
            run_atm_corr(
                Level1_MERIS(
                    self.filename,
                    sline=5774,     scol=299,
                    eline=5774+100, ecol=299+150,
                    ),
                Level2_HDF(target),
                multiprocessing=-1,
                )

    def test_ancillary_nasa(self):

        run_atm_corr(
            Level1_MERIS(
                self.filename,
                sline=5774,     scol=299,
                eline=5774+100, ecol=299+150,
                ancillary=Ancillary_NASA(),
                ),
            Level2('memory'),
            multiprocessing=-1,
            )

    def test_ancillary_era(self):
        raise NotImplementedError


class Test_OLCI(unittest.TestCase):

    filename = '/mfs/user/francois/TESTCASES/OLCI/BALTIC/S3A_OL_1_EFR____20160720T093226_20160720T093526_20160720T113440_0180_006_307_1979_MAR_O_NR_001.SEN3'

    def test1(self):
        run_atm_corr(
            Level1_OLCI(
                self.filename,
                sline=1117,     scol=2871,
                eline=1117+100, ecol=2871+150,
                ),
            Level2('memory'),
            )

    def test_window(self):

        l2_a = run_atm_corr(
                    Level1_OLCI(
                        self.filename,
                        sline=1117,     scol=2871,
                        eline=1117+100, ecol=2871+100,
                        ),
                    Level2('memory'),
                    force_initialization=True,
                    )

        l2_b = run_atm_corr(
                    Level1_OLCI(
                        self.filename,
                        sline=1117+50,     scol=2871+50,
                        eline=1117+50+100, ecol=2871+50+100,
                        blocksize=17,
                        ),
                    Level2('memory'),
                    force_initialization=True,
                    multiprocessing=-1,
                    )
        assert_allclose(
                l2_a.Rw[50:60,50:60,:],
                l2_b.Rw[:10,:10,:]
                )


    def test_external_mask_numpy(self):

        from polymer.common import L2FLAGS

        mask = np.random.randn(100, 150) > 0
        l2 = run_atm_corr(
                    Level1_OLCI(
                        self.filename,
                        sline=1117,     scol=2871,
                        eline=1117+100, ecol=2871+150,
                        ),
                    Level2('memory'),
                    external_mask=mask,
                    )

        assert_equal(l2.bitmask & L2FLAGS['EXTERNAL_MASK'] != 0,
                     mask)

    def test_external_mask_hdf(self):
        from pyhdf.SD import SD, SDC
        from polymer.common import L2FLAGS

        with tempfile.TemporaryDirectory() as tmpdir:
            maskfile = os.path.join(tmpdir, 'mask.hdf')

            # write mask
            shp = (100, 150)
            mask = np.random.randn(*shp) > 0
            mask = mask.astype('int32')

            hdf = SD(maskfile, SDC.WRITE | SDC.CREATE)
            sds = hdf.create('mask', SDC.INT32, shp)
            sds[:] = mask[:]
            sds.endaccess()
            hdf.end()

            l2 = run_atm_corr(
                        Level1_OLCI(
                            self.filename,
                            sline=1117,     scol=2871,
                            eline=1117+100, ecol=2871+150,
                            ),
                        Level2('memory'),
                        external_mask=maskfile,
                        )

            assert_equal(l2.bitmask & L2FLAGS['EXTERNAL_MASK'] != 0,
                         mask)


    def test_external_mask_failure(self):
        ''' test that an exception is raised when providing a wrong mask size '''

        with self.assertRaises(AssertionError):
            mask = np.random.randn(100, 50) > 0
            mask[:,::2] = 1
            run_atm_corr(
                        Level1_OLCI(
                            self.filename,
                            sline=1117,     scol=2871,
                            eline=1117+100, ecol=2871+150,
                            ),
                        Level2('memory'),
                        external_mask=mask,
                        )



class Test_VIIRS(unittest.TestCase):

    def test1(self):
        pass


class Test_MSI(unittest.TestCase):
    '''
    Test Sentinel 2 MSI processing
    '''
    filename = '/mfs/user/francois/TESTCASES/S2A_MSI/Garonne/S2A_OPER_PRD_MSIL1C_PDMC_20160504T225644_R094_V20160504T105917_20160504T105917.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160504T163055_A004524_T30TXR_N02.02'

    def test_window(self):
        ''' test consistency of window processing '''
        l2_a = run_atm_corr(
                  Level1_MSI(
                      self.filename,
                      sline=968,    scol=485,
                      eline=968+100,ecol=485+100,
                      ),
                  Level2('memory'),
                  force_initialization=True,
                  )
        l2_b = run_atm_corr(
                  Level1_MSI(
                      self.filename,
                      sline=968+50,    scol=485+50,
                      eline=968+50+100,ecol=485+50+100,
                      ),
                  Level2('memory'),
                  force_initialization=True,
                  )

        assert_allclose(
                l2_a.Rw[50:60,50:60,:],
                l2_b.Rw[:10,:10,:]
                )


class Test_Level1_Subsetted_NETCDF(unittest.TestCase):

    def test_meris(self):

        run_atm_corr(
                Level1_NETCDF('/mfs/proj/CGLOPS-LAKES/from_Stefan/subsetted_products/subset_0_of_MER_FRS_1PPEPA20080806_094519_000005122071_00022_33643_2741.nc'),
                Level2('memory'),
                multiprocessing=-1,
                )

    def test_olci(self):

        run_atm_corr(
                Level1_NETCDF('/mfs/proj/CGLOPS-LAKES/from_Stefan/subsetted_products/subset_0_of_S3A_OL_1_EFR____20170704T122949_20170704T123249_20170705T182000_0179_019_280_3419_MAR_O_NT_002.nc'),
                Level2('memory'),
                multiprocessing=-1,
                )

class Test_Ancillary_ERA(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
