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
    var numThreads: Int = 4,
    var currentDelegate: Int = 2,
    val modelName: String = "mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite",
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) {
    private var imageClassifier: ImageClassifier? = null

    init {
        setupImageClassifier()
    }

    fun clearImageClassifier() {
        //use if you change the threshold, maxResult, threads, or delegates.
        imageClassifier = null
    }

    private fun setupImageClassifier() {

        val baseOptionsBuilder = BaseOptions.builder()
        baseOptionsBuilder.setDelegate(Delegate.CPU)
        baseOptionsBuilder.setModelAssetPath(modelName)

        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setBaseOptions(baseOptionsBuilder.build())
            .setResultListener { result, image ->
                val finishTimeMs = SystemClock.uptimeMillis()
                val inferenceTime = finishTimeMs - result.timestampMs()
                imageClassifierListener?.onResults(result, inferenceTime)
            }
            .setErrorListener { error ->
                imageClassifierListener?.onError(error.message ?: "An unknown error has occurred")
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

    fun classify(image: ImageProxy) {
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

        val startTime = SystemClock.uptimeMillis()

        val mpImage = BitmapImageBuilder(bitmapBuffer).build()

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setRotationDegrees(image.imageInfo.rotationDegrees)
            .build()

        imageClassifier?.classifyAsync(mpImage, imageProcessingOptions, startTime)
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: ImageClassifierResult?,
            inferenceTime: Long
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2

        private const val TAG = "ImageClassifierHelper"
    }
}