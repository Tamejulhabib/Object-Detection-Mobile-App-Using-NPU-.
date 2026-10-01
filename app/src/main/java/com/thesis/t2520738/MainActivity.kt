package com.thesis.t2520738
import android.Manifest
import android.app.ActivityManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.os.Process
import android.provider.Settings
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
class MainActivity:AppCompatActivity(){
 private lateinit var det:YoloDetector;private lateinit var ov:OverlayView;private lateinit var mon:TextView;private lateinit var preview:PreviewView;private lateinit var videoImage:ImageView;private val ex=Executors.newSingleThreadExecutor();private var last=System.nanoTime();private var cpuPrev=Process.getElapsedCpuTime();private var wallPrev=System.currentTimeMillis();private var accel="OFF"
 override fun onCreate(b:Bundle?){super.onCreate(b);setContentView(R.layout.activity_main);det=YoloDetector(this);det.setMode(false);ov=findViewById(R.id.overlay);mon=findViewById(R.id.monitor);preview=findViewById(R.id.preview);videoImage=findViewById(R.id.videoImage)
  findViewById<Button>(R.id.cpuButton).setOnClickListener{ex.execute{det.setMode(false);accel="OFF"}}
  findViewById<Button>(R.id.npuButton).setOnClickListener{ex.execute{try{accel=det.setMode(true)}catch(e:Exception){accel="NNAPI FAILED: ${e.message}";det.setMode(false)}}}
  findViewById<Button>(R.id.cameraButton).setOnClickListener{startCamera()};findViewById<Button>(R.id.videoButton).setOnClickListener{pickVideo()}; if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)ActivityCompat.requestPermissions(this,arrayOf(Manifest.permission.CAMERA),7) else startCamera() }
 private fun startCamera(){videoImage.visibility=View.GONE;preview.visibility=View.VISIBLE;val f=ProcessCameraProvider.getInstance(this);f.addListener({val p=f.get();val pr=Preview.Builder().build().also{it.setSurfaceProvider(preview.surfaceProvider)};val an=ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();an.setAnalyzer(ex){im->val bm=im.toBitmap();runDetection(bm);im.close()};p.unbindAll();p.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,pr,an)},ContextCompat.getMainExecutor(this))}
 private fun runDetection(b:Bitmap){val(r,ms)=det.detect(b);val now=System.nanoTime();val fps=1e9/(now-last);last=now;val cpuNow=Process.getElapsedCpuTime();val wallNow=System.currentTimeMillis();val cpu=((cpuNow-cpuPrev).toDouble()/(wallNow-wallPrev).coerceAtLeast(1)*100).coerceIn(0.0,100.0);cpuPrev=cpuNow;wallPrev=wallNow;runOnUiThread{ov.setDetections(r,640,640);mon.text="MODE: ${det.mode}   NPU: $accel   CPU: %.1f%%   FPS: %.1f   INFERENCE: %.1f ms   OBJECTS: %d".format(cpu,fps,ms,r.size)}}
 private fun pickVideo(){val i=Intent(Intent.ACTION_OPEN_DOCUMENT).apply{type="video/*";addCategory(Intent.CATEGORY_OPENABLE)};startActivityForResult(i,9)}
 @Deprecated("Deprecated in Java") override fun onActivityResult(r:Int,c:Int,d:Intent?){super.onActivityResult(r,c,d);if(r==9&&c==RESULT_OK&&d?.data!=null){preview.visibility=View.GONE;videoImage.visibility=View.VISIBLE;ex.execute{val m=MediaMetadataRetriever();m.setDataSource(this,d.data);val dur=m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()?:0;var t=0L;while(t<dur){val b=m.getFrameAtTime(t*1000,MediaMetadataRetriever.OPTION_CLOSEST)?:run{t+=100;continue};runOnUiThread{videoImage.setImageBitmap(b)};runDetection(b);t+=100};m.release()}}}
}
