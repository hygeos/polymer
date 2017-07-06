#include <stdio.h>
#include <stdint.h>

#define MIN(a,b)	(((a) > (b)) ? (b) : (a))
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))


/* ---------------------------------------------------------------------------------------- */
/* windex() - return wavelength index of table which is closest to sensor wavelength        */
/* ---------------------------------------------------------------------------------------- */
// (from SeaDAS)
int windex(float wave, float twave[], int ntwave)
{
    int   iw, index;
    float wdiff;
    float wdiffmin = 99999.;

    for (iw=0; iw<ntwave; iw++) {

        /* break on exact match */
        if (twave[iw] == wave) {
  	    index = iw;
	    break;
	}      

        /* look for closest */
        wdiff = fabs(twave[iw]-wave);
        if (wdiff < wdiffmin) {
  	    wdiffmin = wdiff;
            index = iw;
	}
    }

    return(index); 
}


// (from SeaDAS)
void lspline(float xin [],float yin [],int32_t nin,
             float xout[],float yout[],int32_t nout)
{
    float  a, b, x;
    int32_t   i, j;

    /* Can't interpolate if only 1 input point */

    if (nin < 2) {
        for (i=0; i<nout; i++)
	    yout[i] = yin[0];
	return;
    }

    j = 0;

    for (i=0; i<nout; i++) {

        while (xin[j] < xout[i]) {
  	    j = j+1;
            if (j >= nin) {
  	        j = nin-1;
                break;
	    }
	}

        if (xout[i] == xin[j]) {
 	    yout[i] = yin[j];
        } else {
  	    if (j < 1) j=1;
	    a = (yin[j]-yin[j-1])/(xin[j]-xin[j-1]);
            b = yin[j-1];
            x = xout[i] - xin[j-1];
            yout[i] = a*x + b;          
	}
    }

    return;
}


// (from SeaDAS)
float linterp(float xin [],float yin [],int32_t nin,float xout)
{
    float yout;
    lspline(xin,yin,nin,&xout,&yout,1);
    return(yout);
}



/* ---------------------------------------------------------------------------- */
/* fresnel_sol() - effects of the air-sea transmittance for solar path          */
/*                                                                              */
/* Description:                                                                 */
/*   This computes the correction factor on normalized water-leaving radiance   */
/*   to account for the solar zenith angle effects on the downward irradiance   */
/*   from above the ocean surface to below the surface.                         */
/*   Menghua Wang 9/27/04.                                                      */
/*                                                                              */
/* Added windspeed dependence, December 2004, BAF                               */
/* Modified to return air-sea transmittance as option, December 2008, BAF       */            
/* ---------------------------------------------------------------------------- */
// (from SeaDAS)
void fresnel_sol(float wave[],int32_t nwave,float solz,float ws,float brdf[],int return_tf)
{

#define NTWAVE 6
#define NTWIND 5
#define NBANDS 30 // FIXME, this is not the best

    if (nwave > NBANDS) {
        printf("Error in fresnel_sol, stopping.\n");
        exit(1);
    }

    static float radeg = 180./M_PI;
    static int   firstCall = 1;
    static int   tindx[NBANDS]; 
    static float tf0[NBANDS];
    static float twave [] = {412.,443.,490.,510.,555.,670.};
    static float tsigma[] = {0.0,0.1,0.2,0.3,0.4};

    /* M Wang, personal communication, red-nir iterpolated */
    static float tf0_w[] = {412.,443.,490.,510.,555.,670.,765.,865.};
    static float tf0_v[] = {0.965980,0.968320,0.971040,0.971860,0.973450,0.977513,0.980870,0.984403};

    static float c[NTWIND][4][NTWAVE] = {
        { /* ws=0.0 */
	    { -0.0087,-0.0122,-0.0156,-0.0163,-0.0172,-0.0172 }, /* a */
	    {  0.0638, 0.0415, 0.0188, 0.0133, 0.0048,-0.0003 }, /* b */
	    { -0.0379,-0.0780,-0.1156,-0.1244,-0.1368,-0.1430 }, /* c */
	    { -0.0311,-0.0427,-0.0511,-0.0523,-0.0526,-0.0478 }  /* d */
	},
        { /* ws=1.9 */
	    { -0.0011,-0.0037,-0.0068,-0.0077,-0.0090,-0.0106 }, /* a */
	    {  0.0926, 0.0746, 0.0534, 0.0473, 0.0368, 0.0237 }, /* b */
	    { -5.3E-4,-0.0371,-0.0762,-0.0869,-0.1048,-0.1260 }, /* c */
	    { -0.0205,-0.0325,-0.0438,-0.0465,-0.0506,-0.0541 }  /* d */
	},
        { /* ws=7.5 */
	    {  6.8E-5,-0.0018,-0.0011,-0.0012,-0.0015,-0.0013 }, /* a */
	    {  0.1150, 0.1115, 0.1075, 0.1064, 0.1044, 0.1029 }, /* b */
	    {  0.0649, 0.0379, 0.0342, 0.0301, 0.0232, 0.0158 }, /* c */
	    {  0.0065,-0.0039,-0.0036,-0.0047,-0.0062,-0.0072 }  /* d */
	},
        { /* ws=16.9 */
	    { -0.0088,-0.0097,-0.0104,-0.0106,-0.0110,-0.0111 }, /* a */
	    {  0.0697, 0.0678, 0.0657, 0.0651, 0.0640, 0.0637 }, /* b */
	    {  0.0424, 0.0328, 0.0233, 0.0208, 0.0166, 0.0125 }, /* c */
	    {  0.0047, 0.0013,-0.0016,-0.0022,-0.0031,-0.0036 }  /* d */
	},
        { /* ws=30 */
	    { -0.0081,-0.0089,-0.0096,-0.0098,-0.0101,-0.0104 }, /* a */
	    {  0.0482, 0.0466, 0.0450, 0.0444, 0.0439, 0.0434 }, /* b */
	    {  0.0290, 0.0220, 0.0150, 0.0131, 0.0103, 0.0070 }, /* c */
	    {  0.0029, 0.0004,-0.0017,-0.0022,-0.0029,-0.0033 }  /* d */
	}
    };

    float x, x2, x3, x4;
    int is,is1,is2;
    int iw, i;
    float sigma;
    float slp;
    float brdf1, brdf2;

    /* on first call, find closest table entry to each input wavelength */
    if (firstCall) {
        firstCall = 1;
        for (iw=0; iw<nwave; iw++) {
	  tindx[iw] = windex(wave[iw],twave,NTWAVE);
          tf0  [iw] = linterp(tf0_w,tf0_v,8,wave[iw]);
	}
    }

    sigma = 0.0731*sqrt(MAX(ws,0.0));

    x  = log(cos(MIN(solz,80.0)/radeg));
    x2 = x*x;
    x3 = x*x2;
    x4 = x*x3;

    /* find bracketing table winds */
    for (is=0; is<NTWIND; is++)
      if (tsigma[is] > sigma)
	break;
    is2 = MIN(is,NTWIND-1);
    is1 = is2-1;
    slp = (sigma - tsigma[is1])/(tsigma[is2]-tsigma[is1]);

    /* compute at bounding windspeeds and interpolate */
    for (iw=0; iw<nwave; iw++) {
        i = tindx[iw];
        brdf1 = 1.0 + c[is1][0][i]*x + c[is1][1][i]*x2 
                    + c[is1][2][i]*x3 + c[is1][3][i]*x4;
        brdf2 = 1.0 + c[is2][0][i]*x + c[is2][1][i]*x2 
                    + c[is2][2][i]*x3 + c[is2][3][i]*x4;
        brdf[iw] = brdf1 + slp*(brdf2-brdf1);
        if (return_tf != 0) 
	    brdf[iw] = tf0[iw]/brdf[iw];
    }
}
