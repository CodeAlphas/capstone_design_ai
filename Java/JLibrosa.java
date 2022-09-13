package ai.kitt.snowboy.modelUtil;

import java.io.File;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class JLibrosa {
    private int BUFFER_SIZE = 4096;
    private int noOfFrames = -1;
    private int sampleRate = 16000;
    private int noOfChannels = -1;
    private double fMax = 44100 / 2.0;
    private double fMin = 0.0;
    private int n_fft = 2048;
    private int hop_length = 512;
    private int n_mels = 128;

    public double getfMax() {
        return fMax;
    }
    public double getfMin() {
        return fMin;
    }
    public int getN_fft() {
        return n_fft;
    }
    public int getHop_length() {
        return hop_length;
    }
    public int getN_mels() {
        return n_mels;
    }
    public int getNoOfChannels() {
        return noOfChannels;
    }
    public void setNoOfChannels(int noOfChannels) {
        this.noOfChannels = noOfChannels;
    }
    public JLibrosa() { }
    public int getNoOfFrames() {
        return noOfFrames;
    }
    public void setNoOfFrames(int noOfFrames) {
        this.noOfFrames = noOfFrames;
    }
    public int getSampleRate() {
        return sampleRate;
    }
    public void setSampleRate(int sampleRate) {
        this.sampleRate = sampleRate;
        this.fMax = sampleRate/2.0;
    }

    private float[][] readMagnitudeValuesFromFile(String path, int sampleRate, int readDurationInSeconds)
            throws IOException, ai.kitt.snowboy.modelUtil.WavFileException, FileFormatNotSupportedException {

        if(!path.endsWith(".wav")) {
            throw new FileFormatNotSupportedException("File format not supported. jLibrosa currently supports audio processing of only .wav files");
        }

        File sourceFile = new File(path);
        WavFile wavFile = null;

        wavFile = WavFile.openWavFile(sourceFile);
        int mNumFrames = (int) (wavFile.getNumFrames());
        int mSampleRate = (int) wavFile.getSampleRate();
        int mChannels = wavFile.getNumChannels();

        this.setNoOfChannels(mChannels);
        this.setNoOfFrames(mNumFrames);
        this.setSampleRate(mSampleRate);

        if (readDurationInSeconds != -1) {
            mNumFrames = readDurationInSeconds * mSampleRate;
        }

        if (sampleRate != -1) {
            mSampleRate = sampleRate;
        }

        float[][] buffer = new float[mChannels][mNumFrames];
        int frameOffset = 0;
        int loopCounter = ((mNumFrames * mChannels) / BUFFER_SIZE) + 1;
        for (int i = 0; i < loopCounter; i++) {
            frameOffset = wavFile.readFrames(buffer, mNumFrames, frameOffset);
        }

        if(wavFile != null) {
            wavFile.close();
        }

        return buffer;
    }

    public float[][] generateMFCCFeatures(float[] magValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length) {

        AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();

        mfccConvert.setN_mfcc(nMFCC);
        mfccConvert.setN_mels(n_mels);
        mfccConvert.setHop_length(hop_length);

        if(mSampleRate==-1) {
            mSampleRate = this.getSampleRate();
        }

        mfccConvert.setSampleRate(mSampleRate);
        mfccConvert.setN_mfcc(nMFCC);
        float [] mfccInput = mfccConvert.extractMFCCFeatures(magValues);

        int nFFT = mfccInput.length / nMFCC;
        float[][] mfccValues = new float[nMFCC][nFFT];

        for (int i = 0; i < nFFT; i++) {
            int indexCounter = i * nMFCC;
            int rowIndexValue = i % nFFT;
            for (int j = 0; j < nMFCC; j++) {
                mfccValues[j][rowIndexValue] = mfccInput[indexCounter];
                indexCounter++;
            }
        }
        return mfccValues;
    }

    public float[][] generateMFCCFeatures(float[] magValues, int mSampleRate, int nMFCC) {
        float[][] mfccValues = this.generateMFCCFeatures(magValues, mSampleRate, nMFCC, this.n_fft, this.n_mels, this.hop_length);
        return mfccValues;
    }

    public float [] generateMeanMFCCFeatures(float[][] mfccValues, int nMFCC, int nFFT) {

        float [] meanMFCCValues = new float[nMFCC];
        for (int i=0; i<mfccValues.length; i++) {

            float [] floatArrValues = mfccValues[i];
            DoubleStream ds = IntStream.range(0, floatArrValues.length).mapToDouble(k -> floatArrValues[k]);

            double avg = DoubleStream.of(ds.toArray()).average().getAsDouble();
            float floatVal = (float)avg;
            meanMFCCValues[i] = floatVal;
        }
        return meanMFCCValues;
    }

    public float[][] generateMelSpectroGram(float[] yValues, int mSampleRate, int n_fft, int n_mels, int hop_length){
        AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
        mfccConvert.setSampleRate(mSampleRate);
        mfccConvert.setN_fft(n_fft);
        mfccConvert.setN_mels(n_mels);
        mfccConvert.setHop_length(hop_length);
        float [][] melSVal = mfccConvert.melSpectrogramWithComplexValueProcessing(yValues);
        return melSVal;
    }

    public float [] generateMeanMelSpectroGram(float[][] melSVal, int nMel, int nFFT) {
        float [] meanMELValues = new float[nMel];
        for (int i=0; i<melSVal.length; i++) {

            float [] floatArrValues = melSVal[i];
            DoubleStream ds = IntStream.range(0, floatArrValues.length).mapToDouble(k -> floatArrValues[k]);

            double avg = DoubleStream.of(ds.toArray()).average().getAsDouble();
            float floatVal = (float)avg;
            meanMELValues[i] = floatVal;
        }
        return meanMELValues;
    }

    public float[][] generateChromaSpectroGram(float[] yValues, int mSampleRate, int n_fft, int n_mels, int hop_length){
        AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
        mfccConvert.setSampleRate(mSampleRate);
        mfccConvert.setN_fft(n_fft);
        mfccConvert.setN_mels(n_mels);
        mfccConvert.setHop_length(hop_length);
        float [][] melSVal = mfccConvert.chromaSpectrogramWithComplexValueProcessing(yValues);
        return melSVal;
    }

    public float[][] generateChromaSpectroGram(float[] yValues, int mSampleRate) {
        float[][] melSVal = this.generateChromaSpectroGram(yValues, mSampleRate, this.n_fft, this.n_mels, this.hop_length);
        return melSVal;
    }

    public float [] generateMeanChromaSpectroGram(float[][] melSVal, int nMel, int nFFT) {
        float [] meanMELValues = new float[nMel];
        for (int i=0; i<melSVal.length; i++) {

            float [] floatArrValues = melSVal[i];
            DoubleStream ds = IntStream.range(0, floatArrValues.length).mapToDouble(k -> floatArrValues[k]);

            double avg = DoubleStream.of(ds.toArray()).average().getAsDouble();
            float floatVal = (float)avg;
            meanMELValues[i] = floatVal;
        }
        return meanMELValues;
    }

    public float[] loadAndRead(String path, int sampleRate, int readDurationInSeconds)
            throws IOException, ai.kitt.snowboy.modelUtil.WavFileException, FileFormatNotSupportedException {

        float[][] magValueArray = readMagnitudeValuesFromFile(path, sampleRate, readDurationInSeconds);

        DecimalFormat df = new DecimalFormat("#.#####");
        df.setRoundingMode(RoundingMode.CEILING);

        int mNumFrames = this.getNoOfFrames();
        int mChannels = this.getNoOfChannels();
        float[] meanBuffer = new float[mNumFrames];

        for (int q = 0; q < mNumFrames; q++) {
            double frameVal = 0;
            for (int p = 0; p < mChannels; p++) {
                frameVal = frameVal + magValueArray[p][q];
            }
            meanBuffer[q] = Float.parseFloat(df.format(frameVal / mChannels));
        }
        return meanBuffer;
    }
}