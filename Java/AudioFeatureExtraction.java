package ai.kitt.snowboy.modelUtil;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.Arrays;

public class AudioFeatureExtraction {
    private int n_mfcc = 40;
    private double sampleRate = 16000.0;
    private double fMax = sampleRate / 2.0;
    private double fMin = 0.0;
    private int n_fft = 2048;
    private int hop_length = 512;
    private int n_mels = 128;

    public void setSampleRate(double sampleRateVal) {
        sampleRate = sampleRateVal;
        this.fMax = this.sampleRate/2.0;
    }

    public double getfMax() {
        return fMax;
    }
    public void setfMax(double fMax) {
        this.fMax = fMax;
    }
    public double getfMin() {
        return fMin;
    }
    public void setfMin(double fMin) {
        this.fMin = fMin;
    }
    public int getN_fft() {
        return n_fft;
    }
    public void setN_fft(int n_fft) {
        this.n_fft = n_fft;
    }
    public int getHop_length() {
        return hop_length;
    }
    public void setHop_length(int hop_length) {
        this.hop_length = hop_length;
    }
    public int getN_mels() {
        return n_mels;
    }
    public void setN_mels(int n_mels) {
        this.n_mels = n_mels;
    }
    public int getN_mfcc() {
        return n_mfcc;
    }
    public double getSampleRate() {
        return sampleRate;
    }
    public void setN_mfcc(int n_mfccVal) {
        n_mfcc = n_mfccVal;
    }

    public static double logB(double x, double base) {
        return Math.log(x) / Math.log(base);
    }
    public double log10(double value) {
        return Math.log(value) / Math.log(10);
    }

    public float[] extractMFCCFeatures(float[] doubleInputBuffer) {
        final double[][] mfccResult = dctMfcc(doubleInputBuffer);
        return finalshape(mfccResult);
    }

    private float[] finalshape(double[][] mfccSpecTro) {
        float[] finalMfcc = new float[mfccSpecTro[0].length * mfccSpecTro.length];
        int k = 0;
        for (int i = 0; i < mfccSpecTro[0].length; i++) {
            for (int j = 0; j < mfccSpecTro.length; j++) {
                finalMfcc[k] = (float) mfccSpecTro[j][i];
                k = k + 1;
            }
        }
        return finalMfcc;
    }

    private double[][] dctMfcc(float[] y) {
        final double[][] specTroGram = powerToDb(melSpectrogram(y));
        final double[][] dctBasis = dctFilter(n_mfcc, n_mels);
        double[][] mfccSpecTro = new double[n_mfcc][specTroGram[0].length];
        for (int i = 0; i < n_mfcc; i++) {
            for (int j = 0; j < specTroGram[0].length; j++) {
                for (int k = 0; k < specTroGram.length; k++) {
                    mfccSpecTro[i][j] += dctBasis[i][k] * specTroGram[k][j];
                }
            }
        }
        return mfccSpecTro;
    }

    public double[][] melSpectrogram(float[] y) {
        double[][] melBasis = melFilter();
        double[][] spectro = extractSTFTFeatures(y);
        double[][] melS = new double[melBasis.length][spectro[0].length];
        for (int i = 0; i < melBasis.length; i++) {
            for (int j = 0; j < spectro[0].length; j++) {
                for (int k = 0; k < melBasis[0].length; k++) {
                    melS[i][j] += melBasis[i][k] * spectro[k][j];
                }
            }
        }
        return melS;
    }

    public float [][] melSpectrogramWithComplexValueProcessing(float[] y) {

        Complex[][] spectro = extractSTFTFeaturesAsComplexValues(y);
        double[][] spectroAbsVal = new double[spectro.length][spectro[0].length];

        for(int i=0;i<spectro.length;i++) {
            for(int j=0;j<spectro[0].length;j++) {
                Complex complexVal = spectro[i][j];
                double spectroDblVal = Math.sqrt((Math.pow(complexVal.getReal(), 2) + Math.pow(complexVal.getImaginary(), 2)));
                spectroAbsVal[i][j] = Math.pow(spectroDblVal,2);
            }
        }

        double[][] melBasis = melFilter();
        float[][] melS = new float[melBasis.length][spectro[0].length];
        for (int i = 0; i < melBasis.length; i++) {
            for (int j = 0; j < spectro[0].length; j++) {
                for (int k = 0; k < melBasis[0].length; k++) {
                    melS[i][j] += melBasis[i][k] * spectroAbsVal[k][j];
                }
            }
        }
        return melS;
    }

    public float [][] chromaSpectrogramWithComplexValueProcessing(float[] y) {

        Complex[][] spectro = extractSTFTFeaturesAsComplexValues(y);
        double[][] S = new double[spectro.length][spectro[0].length];
        for(int i=0;i<spectro.length;i++) {
            for(int j=0;j<spectro[0].length;j++) {
                Complex complexVal = spectro[i][j];
                double spectroDblVal = Math.sqrt((Math.pow(complexVal.getReal(), 2) + Math.pow(complexVal.getImaginary(), 2)));
                S[i][j] = spectroDblVal;
            }
        }

        int n_ftt = 2 * (S.length - 1);
        double[][] chromafb = chromaFilter(n_ftt, 0.0, 12, 16000.0);

        float[][] raw_chroma = new float[chromafb.length][S[0].length];
        for (int i = 0; i < chromafb.length; i++) {
            for (int j = 0; j < S[0].length; j++) {
                for (int k = 0; k < chromafb[0].length; k++) {
                    raw_chroma[i][j] += chromafb[i][k] * S[k][j];
                }
            }
        }

        double threshold = 2.2250738585072014e-308;

        double [][] mag = new double[raw_chroma.length][raw_chroma[0].length];
        for (int i = 0; i < raw_chroma.length; i++){
            for (int j = 0; j < raw_chroma[0].length; j++){
                mag[i][j] = Math.abs(raw_chroma[i][j]);
            }
        }

        double fill_norm = 1;

        double temp1 = Double.MIN_VALUE;
        double length [] = new double [S[0].length];

        for (int i = 0; i < mag[0].length; i++){
            for (int j = 0; j < mag.length; j++) {
                if (mag[j][i] > temp1) {
                    temp1 = mag[j][i];
                }
            }
            length[i] = temp1;
            temp1 = Double.MIN_VALUE;
        }

        float [][] Snorm = new float[raw_chroma.length][raw_chroma[0].length];
        for (int i = 0; i < raw_chroma.length; i++){
            for (int j = 0; j < raw_chroma[0].length; j++){
                Snorm[i][j] = 0;
            }
        }

        for (int i = 0; i < raw_chroma.length; i++){
            for (int j = 0; j < raw_chroma[0].length; j++){
                if (length[j] < threshold) {
                    length[j] = 1;
                }
                Snorm[i][j] = (float)(raw_chroma[i][j] / length[j]);
            }
        }
        return Snorm;
    }

    public double[][] stftMagSpec(double[] y){
        final double[] fftwin = getWindow();
        double[] ypad = new double[n_fft+y.length];
        for (int i = 0; i < n_fft/2; i++){
            ypad[(n_fft/2)-i-1] = y[i+1];
            ypad[(n_fft/2)+y.length+i] = y[y.length-2-i];
        }
        for (int j = 0; j < y.length; j++){
            ypad[(n_fft/2)+j] = y[j];
        }

        final double[][] frame = yFrame(ypad);
        double[][] fftmagSpec = new double[1+n_fft/2][frame[0].length];
        double[] fftFrame = new double[n_fft];

        for (int k = 0; k < frame[0].length; k++){
            int fftFrameCounter=0;

            for (int l =0; l < n_fft; l++){
                fftFrame[l] = fftwin[l]*frame[l][k];
                fftFrameCounter = fftFrameCounter + 1;

            }

            double[] tempConversion = new double[fftFrame.length];
            double[] tempImag = new double[fftFrame.length];

            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);

                for (int i = 0; i < complx.length; i++) {
                    double rr = (complx[i].getReal());

                    double ri = (complx[i].getImaginary());

                    tempConversion[i] = rr * rr + ri * ri;
                    tempImag[i] = ri;
                }

            } catch (IllegalArgumentException e) {
                System.out.println(e);
            }

            double[] magSpec = tempConversion;
            for (int i =0; i < 1+n_fft/2; i++){
                fftmagSpec[i][k] = magSpec[i];
            }
        }
        return fftmagSpec;
    }

    public Complex[][] extractSTFTFeaturesAsComplexValues(float[] y){

        final double[] fftwin = getWindow();
        final double[][] frame = padFrame(y);

        double[][] fftmagSpec = new double[1 + n_fft / 2][frame[0].length];
        double[] fftFrame = new double[n_fft];

        Complex [][] complex2DArray = new Complex[1 + n_fft / 2][frame[0].length];

        for (int k = 0; k < frame[0].length; k++) {
            int fftFrameCounter = 0;

            for (int l = 0; l < n_fft; l++) {
                fftFrame[fftFrameCounter] = fftwin[l] * frame[l][k];
                fftFrameCounter = fftFrameCounter + 1;
            }

            double[] tempConversion = new double[fftFrame.length];
            double[] tempImag = new double[fftFrame.length];

            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);

            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);

                for(int i=0;i<1+n_fft/2;i++) {
                    complex2DArray[i][k] = complx[i];
                }

            } catch (IllegalArgumentException e) {
                System.out.println(e);
            }
        }
        return complex2DArray;
    }

    public double[][] padFrame(float[] yValues){
        double[] ypad = new double[n_fft + yValues.length];
        for (int i = 0; i < n_fft / 2; i++) {
            ypad[(n_fft / 2) - i - 1] = yValues[i + 1];
            ypad[(n_fft / 2) + yValues.length + i] = yValues[yValues.length - 2 - i];
        }
        for (int j = 0; j < yValues.length; j++) {
            ypad[(n_fft / 2) + j] = yValues[j];
        }
        final double[][] frame = yFrame(ypad);
        return frame;
    }

    public Complex[][] extractSTFTFeaturesAsComplexValues1(float[] y){

        double win_length = 2048;
        double hop_length = 512;

        final double[] fftwin = getWindow();
        final double[] frame = pad_reflect(y, (int) (n_fft / 2));
        double[] fftFrame = new double[n_fft];

        Complex [][] complex2DArray = new Complex[1 + n_fft / 2][frame.length];

        for (int k = 0; k < frame.length; k++) {
            int fftFrameCounter = 0;

            for (int l = 0; l < n_fft; l++) {
                fftFrame[fftFrameCounter] = fftwin[l] * frame[k];
                fftFrameCounter = fftFrameCounter + 1;
            }

            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);

            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);

                for(int i=0;i<1+n_fft/2;i++) {
                    complex2DArray[i][k] = complx[i];
                }

            } catch (IllegalArgumentException e) {
                System.out.println(e);
            }
        }
        return complex2DArray;
    }

    public double[] pad_reflect(float[] array1, int pad_width) {
        int a = array1.length;
        System.out.printf("%d", a);
        double[] array = new double[a];
        for (int i = 0; i < a; i++) {
            array[i] = array1[i];
        }

        double[] ret = new double[array.length + pad_width * 2];

        if (array.length == 0) {
            throw new IllegalArgumentException("can't extend empty axis 0 using modes other than 'constant' or 'empty'");
        }

        if (array.length == 1) {
            Arrays.fill(ret, array[0]);
            return ret;
        }

        int pos = 0;
        int dis = -1;
        for (int i = 0; i < pad_width; i++) {
            if (pos == array.length - 1 || pos == 0) {
                dis = -dis;
            }
            pos += dis;
            ret[pad_width - i - 1] = array[pos];
        }

        System.arraycopy(array, 0, ret, pad_width, array.length);

        pos = array.length - 1;
        dis = 1;
        for (int i = 0; i < pad_width; i++) {
            if (pos == array.length - 1 || pos == 0) {
                dis = -dis;
            }
            pos += dis;
            ret[pad_width + array.length + i] = array[pos];
        }
        return ret;
    }

    public double[][] extractSTFTFeatures(float[] y) {

        final double[] fftwin = getWindow();
        final double[][] frame = padFrame(y);
        double[][] fftmagSpec = new double[1 + n_fft / 2][frame[0].length];

        double[] fftFrame = new double[n_fft];

        for (int k = 0; k < frame[0].length; k++) {
            int fftFrameCounter = 0;
            for (int l = 0; l < n_fft; l++) {
                fftFrame[fftFrameCounter] = fftwin[l] * frame[l][k];
                fftFrameCounter = fftFrameCounter + 1;
            }

            double[] tempConversion = new double[fftFrame.length];
            double[] tempImag = new double[fftFrame.length];

            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);

            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);
                for (int i = 0; i < complx.length; i++) {
                    double rr = (complx[i].getReal());

                    double ri = (complx[i].getImaginary());

                    tempConversion[i] = rr * rr + ri*ri;
                    tempImag[i] = ri;
                }

            } catch (IllegalArgumentException e) {
                System.out.println(e);
            }

            double[] magSpec = tempConversion;
            for (int i = 0; i < 1 + n_fft / 2; i++) {
                fftmagSpec[i][k] = magSpec[i];
            }
        }
        return fftmagSpec;
    }

    private double[] getWindow() {
        double[] win = new double[n_fft];
        for (int i = 0; i < n_fft; i++) {
            //win[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / n_fft);
            win[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / (n_fft-1));
        }
        return win;
    }

    private double[][] yFrame(double[] ypad) {
        final int n_frames = 1 + (ypad.length - n_fft) / hop_length;
        double[][] winFrames = new double[n_fft][n_frames];
        for (int i = 0; i < n_fft; i++) {
            for (int j = 0; j < n_frames; j++) {
                winFrames[i][j] = ypad[j * hop_length + i];
            }
        }
        return winFrames;
    }

    private double[][] powerToDb(double[][] melS) {
        double[][] log_spec = new double[melS.length][melS[0].length];
        double maxValue = -100;
        for (int i = 0; i < melS.length; i++) {
            for (int j = 0; j < melS[0].length; j++) {
                double magnitude = Math.abs(melS[i][j]);
                if (magnitude > 1e-10) {
                    log_spec[i][j] = 10.0 * log10(magnitude);
                } else {
                    log_spec[i][j] = 10.0 * (-10);
                }
                if (log_spec[i][j] > maxValue) {
                    maxValue = log_spec[i][j];
                }
            }
        }
        for (int i = 0; i < melS.length; i++) {
            for (int j = 0; j < melS[0].length; j++) {
                if (log_spec[i][j] < maxValue - 80.0) {
                    log_spec[i][j] = maxValue - 80.0;
                }
            }
        }
        return log_spec;
    }

    private double[][] dctFilter(int n_filters, int n_input) {
        double[][] basis = new double[n_filters][n_input];
        double[] samples = new double[n_input];
        for (int i = 0; i < n_input; i++) {
            samples[i] = (1 + 2 * i) * Math.PI / (2.0 * (n_input));
        }
        for (int j = 0; j < n_input; j++) {
            basis[0][j] = 1.0 / Math.sqrt(n_input);
        }
        for (int i = 1; i < n_filters; i++) {
            for (int j = 0; j < n_input; j++) {
                basis[i][j] = Math.cos(i * samples[j]) * Math.sqrt(2.0 / (n_input));
            }
        }
        return basis;
    }

    private double[][] melFilter() {
        final double[] fftFreqs = fftFreq();
        final double[] melF = melFreq(n_mels + 2);

        double[] fdiff = new double[melF.length - 1];
        for (int i = 0; i < melF.length - 1; i++) {
            fdiff[i] = melF[i + 1] - melF[i];
        }

        double[][] ramps = new double[melF.length][fftFreqs.length];
        for (int i = 0; i < melF.length; i++) {
            for (int j = 0; j < fftFreqs.length; j++) {
                ramps[i][j] = melF[i] - fftFreqs[j];
            }
        }

        double[][] weights = new double[n_mels][1 + n_fft / 2];
        for (int i = 0; i < n_mels; i++) {
            for (int j = 0; j < fftFreqs.length; j++) {
                double lowerF = -ramps[i][j] / fdiff[i];
                double upperF = ramps[i + 2][j] / fdiff[i + 1];
                if (lowerF > upperF && upperF > 0) {
                    weights[i][j] = upperF;
                } else if (lowerF > upperF && upperF < 0) {
                    weights[i][j] = 0;
                } else if (lowerF < upperF && lowerF > 0) {
                    weights[i][j] = lowerF;
                } else if (lowerF < upperF && lowerF < 0) {
                    weights[i][j] = 0;
                } else {
                }
            }
        }

        double enorm[] = new double[n_mels];
        for (int i = 0; i < n_mels; i++) {
            enorm[i] = 2.0 / (melF[i + 2] - melF[i]);
            for (int j = 0; j < fftFreqs.length; j++) {
                weights[i][j] *= enorm[i];
            }
        }
        return weights;
    }

    public double[][] chromaFilter(int n_fft, double tuning, int n_chroma, double sr) {

        double [][] wts = new double[n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                wts[i][j] = 0;
            }
        }

        double frequencies [] = new double[n_fft-1];
        for (int i = 0; i < n_fft-1; i++){
            frequencies[i] = ((sr/(double)n_fft) * (i+1));
        }

        double frqbins [] = new double[n_fft-1];

        double A440 = 440.0 * Math.pow(2.0 , (tuning / 12));

        for (int i = 0; i < n_fft-1; i++){
            frqbins [i] = n_chroma * logB((frequencies[i]/(A440/16)), 2) ;
        }

        double [] frqbins1 = new double [n_fft];
        frqbins1[0] = frqbins[0] - 1.5 * n_chroma;
        for (int i = 1; i < n_fft; i++){
            frqbins1[i] = frqbins [i-1];
        }

        double [] binwidthbins = new double [n_fft];
        for (int i = 0; i < n_fft-1; i++){
            binwidthbins[i] = Math.max(frqbins1[i+1] - frqbins1[i], 1.0);
        }
        binwidthbins[n_fft-1] = 1;

        double [][] D = new double [n_chroma][n_fft];
        for (int i = 0; i < 12; i++){
            for (int j = 0; j < n_fft; j++){
                D[i][j] = frqbins1[j] - i;
            }
        }

        double n_chroma2 = 6.0;

        double [][] D1 = new double [n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                D1[i][j] = ((D[i][j] + n_chroma2 + 10 * 12) % 12) - n_chroma2;
            }
        }

        double [][] wts1 = new double[n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                wts1[i][j] = Math.exp(-0.5 * (Math.pow((2 * D1[i][j] / binwidthbins[j]), 2)));
            }
        }

        double threshold = 2.2250738585072014e-308;

        double [][] mag = new double[n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                mag[i][j] = Math.abs(wts1[i][j]);
            }
        }

        int fill_norm = 1;
        double length [] = new double [n_fft];
        double temp2 = Double.MIN_VALUE;

        for (int i = 0; i < mag[0].length; i++){
            for (int j = 0; j < mag.length; j++) {
                if (mag[j][i] > temp2) {
                    temp2 = mag[j][i];
                }
            }
            length[i] = temp2;
            temp2 = Double.MIN_VALUE;
        }

        double [][] Snorm = new double[n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                Snorm[i][j] = 0;
            }
        }

        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                if (length[j] < threshold) {
                    length[j] = 1.0;
                }
                Snorm[i][j] = wts1[i][j] / length[j];
            }
        }

        for (int i = 0; i < n_chroma; i++){
            for (int j = 0; j < n_fft; j++){
                wts1[i][j] = Snorm[i][j];
            }
        }

        double tile [][] = new double [n_chroma][n_fft];
        for (int i = 0; i < n_chroma; i ++) {
            for (int j = 0; j < n_fft; j++) {
                tile[i][j] = Math.exp(-0.5 * Math.pow(((frqbins1[j] / 12 - 5) / 2), 2));
            }
        }

        for (int i = 0; i < n_chroma; i ++) {
            for (int j = 0; j < n_fft; j++) {
                wts1[i][j] = wts1[i][j] * tile[i][j];
            }
        }

        double temp [][] = new double [12][n_fft];
        for (int i = 0; i < n_chroma; i ++) {
            for (int j = 0; j < n_fft; j++) {
                temp[i][j] = wts1[i][j];
            }
        }

        for (int i = 0; i < n_chroma-3; i ++) {
            for (int j = 0; j < n_fft; j++) {
                wts1[i][j] = temp[i+3][j];
            }
        }

        for (int i = n_chroma-3; i < n_chroma; i ++) {
            for (int j = 0; j < n_fft; j++) {
                wts1[i][j] = temp[i-9][j];
            }
        }

        int wa = 1 + n_fft/2;
        double weight [][] = new double [12][wa];

        for (int i = 0; i < 12; i ++) {
            for (int j = 0; j < wa; j++) {
                weight[i][j] = wts1[i][j];
            }
        }
        return weight;
    }

    private double[] fftFreq() {
        double[] freqs = new double[1 + n_fft / 2];
        for (int i = 0; i < 1 + n_fft / 2; i++) {
            freqs[i] = 0 + (sampleRate / 2) / (n_fft / 2) * i;
        }
        return freqs;
    }

    private double[] melFreq(int numMels) {
        double[] LowFFreq = new double[1];
        double[] HighFFreq = new double[1];
        LowFFreq[0] = fMin;
        HighFFreq[0] = fMax;
        final double[] melFLow = freqToMel(LowFFreq);
        final double[] melFHigh = freqToMel(HighFFreq);
        double[] mels = new double[numMels];
        for (int i = 0; i < numMels; i++) {
            mels[i] = melFLow[0] + (melFHigh[0] - melFLow[0]) / (numMels - 1) * i;
        }
        return melToFreq(mels);
    }

    private double[] melToFreqS(double[] mels) {
        double[] freqs = new double[mels.length];
        for (int i = 0; i < mels.length; i++) {
            freqs[i] = 700.0 * (Math.pow(10, mels[i] / 2595.0) - 1.0);
        }
        return freqs;
    }

    protected double[] freqToMelS(double[] freqs) {
        double[] mels = new double[freqs.length];
        for (int i = 0; i < freqs.length; i++) {
            mels[i] = 2595.0 * log10(1.0 + freqs[i] / 700.0);
        }
        return mels;
    }

    private double[] melToFreq(double[] mels) {
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] freqs = new double[mels.length];

        final double min_log_hz = 1000.0;
        final double min_log_mel = (min_log_hz - f_min) / f_sp;
        final double logstep = Math.log(6.4) / 27.0;

        for (int i = 0; i < mels.length; i++) {
            if (mels[i] < min_log_mel) {
                freqs[i] = f_min + f_sp * mels[i];
            } else {
                freqs[i] = min_log_hz * Math.exp(logstep * (mels[i] - min_log_mel));
            }
        }
        return freqs;
    }

    protected double[] freqToMel(double[] freqs) {
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] mels = new double[freqs.length];

        final double min_log_hz = 1000.0;
        final double min_log_mel = (min_log_hz - f_min) / f_sp;
        final double logstep = Math.log(6.4) / 27.0;

        for (int i = 0; i < freqs.length; i++) {
            if (freqs[i] < min_log_hz) {
                mels[i] = (freqs[i] - f_min) / f_sp;
            } else {
                mels[i] = min_log_mel + Math.log(freqs[i] / min_log_hz) / logstep;
            }
        }
        return mels;
    }

}