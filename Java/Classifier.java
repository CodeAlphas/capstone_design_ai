package ai.kitt.snowboy.modelUtil;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class Classifier {
    private static final String LOG_TAG = ai.kitt.snowboy.modelUtil.Classifier.class.getSimpleName();
    private static final String MODEL_NAME = "speechemotion.tflite";
    private static final int NUM_CLASSES = 4;

    private final Interpreter.Options options = new Interpreter.Options();
    private final Interpreter mInterpreter;
    private final float[][][] mResult = new float[1][1][NUM_CLASSES];
    private final float[] featurevector = new float[180];


    public Classifier(float[] audioFeatureValues, Activity activity) throws IOException, FileFormatNotSupportedException, ai.kitt.snowboy.modelUtil.WavFileException {
        mInterpreter = new Interpreter(loadModelFile(activity), options);

        ai.kitt.snowboy.modelUtil.JLibrosa jLibrosa = new ai.kitt.snowboy.modelUtil.JLibrosa();

        int nNoOfFrames = jLibrosa.getNoOfFrames();
        int sampleRate = jLibrosa.getSampleRate();
        int noOfChannels = jLibrosa.getNoOfChannels();

        float [][] melSpectrogram = jLibrosa.generateMelSpectroGram(audioFeatureValues, sampleRate, 2048, 128, 256);
        float[] meanMELValues = jLibrosa.generateMeanMelSpectroGram(melSpectrogram, melSpectrogram.length, melSpectrogram[0].length);

        float[][] mfccValues = jLibrosa.generateMFCCFeatures(audioFeatureValues, sampleRate, 40);
        float[] meanMFCCValues = jLibrosa.generateMeanMFCCFeatures(mfccValues, mfccValues.length, mfccValues[0].length);

        float[][] chromaValues = jLibrosa.generateChromaSpectroGram(audioFeatureValues, 16000);
        float[] meanchromaValues = jLibrosa.generateMeanChromaSpectroGram(chromaValues, chromaValues.length, chromaValues[0].length);

        for(int i=0; i<meanMFCCValues.length; i++) {
            featurevector[i] = meanMFCCValues[i];
        }
        for(int i=40; i<52; i++) {
            featurevector[i] = meanchromaValues[i-40];
        }
        for(int i=52; i<180; i++) {
            featurevector[i] = meanMELValues[i-52];
        }
    }

    public Result classify() {

        long startTime = SystemClock.uptimeMillis();

        float [][][] comfeaturevector = new float [1][1][180];

        for (int k = 0; k < 180; k++) {
            comfeaturevector[0][0][k] = featurevector[k];
        }

        mInterpreter.run(comfeaturevector, mResult);

        long endTime = SystemClock.uptimeMillis();
        long timeCost = endTime - startTime;

        Log.v(LOG_TAG, "classify(): result = " + Arrays.toString(mResult[0][0])
                + ", timeCost = " + timeCost);
        return new Result(mResult[0][0], timeCost);
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {

        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        Log.i("my log", "cur dir: success");

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}