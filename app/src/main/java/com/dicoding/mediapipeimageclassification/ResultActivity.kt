package com.dicoding.mediapipeimageclassification

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.core.net.toUri
import com.dicoding.mediapipeimageclassification.databinding.ActivityResultBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifierResult

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var imageClassifierHelper: ImageClassifierHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imageUri = intent.getStringExtra(DATA_IMAGE)?.toUri()

        binding.resultImage.setImageURI(imageUri)

        imageClassifierHelper =
            ImageClassifierHelper(
                context = this,
                runningMode = RunningMode.IMAGE,
                imageClassifierListener = object : ImageClassifierHelper.ClassifierListener {
                    override fun onError(error: String) {
                        runOnUiThread {
                            Toast.makeText(this@ResultActivity, error, Toast.LENGTH_SHORT).show()
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
                                                "%.2f",
                                                it.score()
                                            ).trim()
                                        }
                                    binding.tvResult.text = displayResult
                                }
                            }
                        }
                    }
                })

        imageUri?.let { uri ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, imageUri)
                ImageDecoder.decodeBitmap(source)
            } else {
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }.copy(Bitmap.Config.ARGB_8888, true)?.let { bitmap ->
                imageClassifierHelper.classifyImage(bitmap)
            }
        }
    }

    companion object{
        const val DATA_IMAGE = "data_image"
    }
}