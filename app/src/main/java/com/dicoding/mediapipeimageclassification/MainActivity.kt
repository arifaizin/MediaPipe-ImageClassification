package com.dicoding.mediapipeimageclassification

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.dicoding.mediapipeimageclassification.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifierResult
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageClassifierHelper: ImageClassifierHelper
    private lateinit var binding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null


    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            val message =
                if (isGranted) "Camera permission granted" else "Camera permission rejected"
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        requestPermissionLauncher.launch(Manifest.permission.CAMERA)

        startCamera()

        binding.captureImage.setOnClickListener {
            takePhoto()
        }
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@MainActivity, "Gagal mengambil gambar.", Toast.LENGTH_SHORT)
                        .show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = output.savedUri

                    val intent = Intent(this@MainActivity, ResultActivity::class.java)
                    intent.putExtra(ResultActivity.DATA_IMAGE, savedUri.toString())
                    startActivity(intent)
                }
            }
        )
    }

    private fun startCamera() {

        imageClassifierHelper =
            ImageClassifierHelper(
                context = this,
                runningMode = RunningMode.LIVE_STREAM,
                imageClassifierListener = object : ImageClassifierHelper.ClassifierListener {
                    override fun onError(error: String) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, error, Toast.LENGTH_SHORT).show()
                        }
                    }

                    override fun onResults(results: ImageClassifierResult?, inferenceTime: Long) {
                        runOnUiThread {
                            results?.classificationResult()?.classifications()?.let { it ->
                                println(it)

                                if (it.isNotEmpty() && it[0].categories().isNotEmpty()) {
                                    println(it)
                                    val sortedCategories =
                                        it[0].categories().sortedByDescending { it?.score() }
                                    val displayResult =
                                        sortedCategories.joinToString("\n") {
                                            "${it.categoryName()} " + String.format(
                                                "%.2f%%",
                                                it.score()
                                            ).trim()
                                        }
                                    binding.tvResult.text = displayResult
                                }
                            }
                        }
                    }
                })

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer =
                ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setTargetRotation(binding.viewFinder.display.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    // The analyzer can then be assigned to the instance
                    .also {
                        it.setAnalyzer(Executors.newSingleThreadExecutor()) { image ->
                            // Pass Bitmap and rotation to the image classifier helper for processing and classification
                            imageClassifierHelper.classifyLiveStreamFrame(image)
                        }
                    }

            imageCapture = ImageCapture.Builder().build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                    imageAnalyzer
                )
            } catch (exc: Exception) {
                Toast.makeText(this, "Gagal memunculkan kamera.", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onPause() {
        super.onPause()
        imageClassifierHelper.clearImageClassifier()
    }

    override fun onResume() {
        super.onResume()
        if (imageClassifierHelper.isClosed()) {
            imageClassifierHelper.setupImageClassifier()
        }
    }

    companion object{
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }
}