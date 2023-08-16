package com.dicoding.mediapipeimageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifier
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifierResult

class ImageClassifierHelper(
    var threshold: Float = 0.1f,
    var maxResults: Int = 3,
    var currentDelegate: Delegate = Delegate.GPU,
    val modelName: String = "mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite",
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) {
    private var imageClassifier: ImageClassifier? = null

    init {
        setupImageClassifier()
    }

    fun clearImageClassifier() {
        // Classifier must be closed when creating a new one to avoid returning results to a
        // non-existent object
        imageClassifier?.close()
        //use if you change the threshold, maxResult, threads, or delegates.
        imageClassifier = null
    }

    fun isClosed(): Boolean {
        return imageClassifier == null
    }

    fun setupImageClassifier() {

        val baseOptionsBuilder = BaseOptions.builder()
        baseOptionsBuilder.setDelegate(currentDelegate)
        baseOptionsBuilder.setModelAssetPath(modelName)

        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
            .setRunningMode(runningMode)
            .setBaseOptions(baseOptionsBuilder.build())

        if (runningMode == RunningMode.LIVE_STREAM) {
            optionsBuilder.setResultListener { result, image ->
                val finishTimeMs = SystemClock.uptimeMillis()
                val inferenceTime = finishTimeMs - result.timestampMs()
                imageClassifierListener?.onResults(result, inferenceTime)
            }
            optionsBuilder.setErrorListener { error ->
                imageClassifierListener?.onError(
                    error.message ?: "An unknown error has occurred"
                )
            }
        }

        val options = optionsBuilder.build()
        try {
            imageClassifier = ImageClassifier.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            imageClassifierListener?.onError(
                "Image classifier failed to initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    fun classifyLiveStreamFrame(image: ImageProxy) {
        if (imageClassifier == null) {
            setupImageClassifier()
        }

        val bitmapBuffer = Bitmap.createBitmap(
            image.width,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        image.close()

        val frameTime = SystemClock.uptimeMillis()

        val mpImage = BitmapImageBuilder(bitmapBuffer).build()

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setRotationDegrees(image.imageInfo.rotationDegrees)
            .build()

        imageClassifier?.classifyAsync(mpImage, imageProcessingOptions, frameTime)
    }

    fun classifyImage(bitmap: Bitmap) {
        if (imageClassifier == null) {
            setupImageClassifier()
        }

        val mpImage = BitmapImageBuilder(bitmap).build()

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .build()

        val startTime = SystemClock.uptimeMillis()

        imageClassifier?.classify(mpImage, imageProcessingOptions).also { result ->
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            imageClassifierListener?.onResults(result, inferenceTime)
        }

        if (imageClassifier == null) {
            imageClassifierListener?.onError(
                "Image classifier failed to classify."
            )
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: ImageClassifierResult?,
            inferenceTime: Long
        )
    }

    companion object {
        private const val TAG = "ImageClassifierHelper"
    }
}