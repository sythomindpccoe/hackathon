require('dotenv').config();
const express = require('express');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const http = require('http');
const { Server } = require('socket.io');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const sharp = require('sharp');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegStatic = require('ffmpeg-static');

// Set ffmpeg path
ffmpeg.setFfmpegPath(ffmpegStatic);

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

const PORT = process.env.PORT || 3000;
const ROBOFLOW_ENDPOINT = process.env.ROBOFLOW_ENDPOINT;
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY;

if (!ROBOFLOW_ENDPOINT || !ROBOFLOW_API_KEY) {
  console.warn('WARNING: ROBOFLOW_ENDPOINT or ROBOFLOW_API_KEY missing in .env');
}

// Alert threshold system
let ALERT_THRESHOLD = 5; // Default threshold
const ALERT_COOLDOWN = 5 * 60 * 1000; // 5 minutes cooldown
let lastAlertTime = 0;

// views + static
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));

// ensure uploads dir exists
const uploadDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// multer for uploads (images and videos)
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^\w.-]/g, '_');
    cb(null, `${Date.now()}-${safe}`);
  }
});
const upload = multer({ 
  storage, 
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB for videos
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/') || file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image and video files are allowed'));
    }
  }
});

// ROUTES
app.get('/', (req, res) => {
  res.render('input', { title: 'Crowd Count - Input' });
});

app.get('/monitor', (req, res) => {
  res.render('monitor', { title: 'Crowd Count - Monitor' });
});

app.get('/upload', (req, res) => {
  res.render('upload', { title: 'Crowd Count - Upload' });
});

app.get('/video', (req, res) => {
  res.render('video', { title: 'Crowd Count - Video Upload' });
});

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, error: 'No image uploaded' });

    const buf = fs.readFileSync(req.file.path);
    const base64 = buf.toString('base64');

    const resp = await axios.post(
      ROBOFLOW_ENDPOINT,
      { api_key: ROBOFLOW_API_KEY, inputs: { image: { type: 'base64', value: base64 }, confidence: 0.3} },
      { headers: { 'Content-Type': 'application/json' }, timeout: 30000 }
    );

    const output = resp.data.outputs?.[0] ?? {};
    const predictions = output.predictions?.predictions || [];
    const count = output.count_objects ?? predictions.length;
    
    // Check threshold for API responses
    if (count >= ALERT_THRESHOLD && Date.now() - lastAlertTime > ALERT_COOLDOWN) {
      lastAlertTime = Date.now();
      console.log(`ALERT: Threshold exceeded (${count} people)`);
    }
    
    // Generate dot annotations for uploaded images
    const annotatedImageBuffer = await drawDotsOnImage(buf, predictions);
    const annotatedImageBase64 = annotatedImageBuffer.toString('base64');
    
    res.json({
      success: true,
      originalImage: `/uploads/${req.file.filename}`,
      count: count,
      annotatedImage: annotatedImageBase64,
      predictions: predictions
    });
  } catch (err) {
    console.error('Predict error:', err?.response?.data || err.message);
    res.status(500).json({ success: false, error: err.message || 'Prediction failed' });
  }
});

// Enhanced function to create annotated frame with count overlay
async function createAnnotatedFrameWithCount(frameBuffer, predictions, frameNumber, timestamp) {
  try {
    const count = predictions.length;
    
    // Load the original image
    const image = await sharp(frameBuffer).toBuffer();
    const img = await loadImage(image);
    
    // Create canvas with extra width for count panel
    const originalWidth = img.width;
    const originalHeight = img.height;
    const panelWidth = 300;
    const totalWidth = originalWidth + panelWidth;
    
    const canvas = createCanvas(totalWidth, originalHeight);
    const ctx = canvas.getContext('2d');
    
    // Draw original image on the left
    ctx.drawImage(img, 0, 0);
    
    // Draw dots on people
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
    predictions.forEach(pred => {
      try {
        const x = pred.x;
        const y = pred.y;
        const dotSize = Math.min(15, Math.max(5, Math.sqrt(pred.width * pred.height) / 6));
        
        if (x && y) {
          ctx.beginPath();
          ctx.arc(x, y, dotSize, 0, Math.PI * 2);
          ctx.fill();
          
          // Add confidence text
          ctx.fillStyle = 'white';
          ctx.font = 'bold 12px Arial';
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 2;
          ctx.strokeText(`${Math.round(pred.confidence * 100)}%`, x + dotSize + 2, y);
          ctx.fillText(`${Math.round(pred.confidence * 100)}%`, x + dotSize + 2, y);
          ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        }
      } catch (err) {
        console.warn('Error drawing prediction:', err);
      }
    });
    
    // Draw right panel with dark background
    ctx.fillStyle = 'rgba(30, 30, 30, 0.95)';
    ctx.fillRect(originalWidth, 0, panelWidth, originalHeight);
    
    // Panel styling
    const panelX = originalWidth + 20;
    
    // Title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 24px Arial';
    ctx.fillText('CROWD ANALYSIS', panelX, 40);
    
    // Current count - large and prominent
    ctx.fillStyle = '#00ff41';
    ctx.font = 'bold 48px Arial';
    ctx.fillText(`${count}`, panelX, 120);
    
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Arial';
    ctx.fillText('People Detected', panelX, 140);
    
    // Frame info
    ctx.fillStyle = '#888888';
    ctx.font = '14px Arial';
    ctx.fillText(`Frame: ${frameNumber}`, panelX, 180);
    ctx.fillText(`Time: ${timestamp}`, panelX, 200);
    
    // Detection details
    if (count > 0) {
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 18px Arial';
      ctx.fillText('Detections:', panelX, 240);
      
      ctx.font = '12px Arial';
      let y = 260;
      predictions.slice(0, 8).forEach((pred, i) => {
        const conf = Math.round(pred.confidence * 100);
        ctx.fillStyle = conf > 80 ? '#00ff41' : conf > 60 ? '#ffaa00' : '#ff6b6b';
        ctx.fillText(`${i + 1}. ${conf}% confidence`, panelX, y);
        y += 16;
      });
      
      if (predictions.length > 8) {
        ctx.fillStyle = '#888888';
        ctx.fillText(`... and ${predictions.length - 8} more`, panelX, y);
      }
    }
    
    // Status indicator
    ctx.fillStyle = count > 10 ? '#ff4444' : count > 5 ? '#ffaa00' : '#00ff41';
    ctx.fillRect(panelX, originalHeight - 60, 20, 20);
    
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    const status = count > 10 ? 'HIGH DENSITY' : count > 5 ? 'MODERATE' : 'LOW DENSITY';
    ctx.fillText(status, panelX + 30, originalHeight - 45);
    
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } catch (err) {
    console.error('Error creating annotated frame:', err);
    return frameBuffer; // Return original if processing fails
  }
}

// Video processing route
app.post('/predict-video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, error: 'No video uploaded' });

    const videoPath = req.file.path;
    const videoFilename = req.file.filename;
    const outputVideoName = `annotated_${videoFilename.replace(/\.[^/.]+$/, ".mp4")}`;
    const outputVideoPath = path.join(uploadDir, outputVideoName);
    const framesDir = path.join(uploadDir, 'frames', videoFilename.replace(/\.[^/.]+$/, ""));
    const annotatedFramesDir = path.join(uploadDir, 'annotated_frames', videoFilename.replace(/\.[^/.]+$/, ""));
    
    // Create directories
    [framesDir, annotatedFramesDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    // Extract frames from video (every 0.5 seconds)
    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .fps(2) // Extract 2 frames per second
        .output(path.join(framesDir, 'frame_%04d.jpg'))
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    // Get all frame files
    const frameFiles = fs.readdirSync(framesDir)
      .filter(file => file.endsWith('.jpg'))
      .sort();

    const frameResults = [];
    let frameCount = 0;
    const allCounts = [];

    // Process each frame
    for (const frameFile of frameFiles) {
      const framePath = path.join(framesDir, frameFile);
      const frameBuffer = fs.readFileSync(framePath);
      const base64 = frameBuffer.toString('base64');

      try {
        const resp = await axios.post(
          ROBOFLOW_ENDPOINT,
          { api_key: ROBOFLOW_API_KEY, inputs: { image: { type: 'base64', value: base64 }, confidence: 0.3} },
          { headers: { 'Content-Type': 'application/json' }, timeout: 30000 }
        );

        const output = resp.data.outputs?.[0] ?? {};
        const predictions = output.predictions?.predictions || [];
        const count = output.count_objects ?? predictions.length;
        const timestamp = (frameCount * 0.5).toFixed(1) + 's';

        // Create enhanced annotated frame with count panel
        const annotatedBuffer = await createAnnotatedFrameWithCount(frameBuffer, predictions, frameCount + 1, timestamp);
        const annotatedPath = path.join(annotatedFramesDir, frameFile);
        fs.writeFileSync(annotatedPath, annotatedBuffer);

        frameResults.push({
          frameNumber: frameCount++,
          originalFrame: `/uploads/frames/${path.basename(framesDir)}/${frameFile}`,
          annotatedFrame: `/uploads/annotated_frames/${path.basename(annotatedFramesDir)}/${frameFile}`,
          count: count,
          predictions: predictions,
          timestamp: timestamp
        });

        allCounts.push(count);

        // Add delay to avoid overwhelming the API
        await new Promise(resolve => setTimeout(resolve, 300));
        
      } catch (apiError) {
        console.error('API error for frame:', frameFile, apiError.message);
        // Create a frame with error message
        const timestamp = (frameCount * 0.5).toFixed(1) + 's';
        const errorBuffer = await createAnnotatedFrameWithCount(
          frameBuffer, 
          [], 
          frameCount + 1, 
          timestamp + ' (ERROR)'
        );
        const annotatedPath = path.join(annotatedFramesDir, frameFile);
        fs.writeFileSync(annotatedPath, errorBuffer);
        
        frameResults.push({
          frameNumber: frameCount++,
          originalFrame: `/uploads/frames/${path.basename(framesDir)}/${frameFile}`,
          annotatedFrame: `/uploads/annotated_frames/${path.basename(annotatedFramesDir)}/${frameFile}`,
          count: 0,
          predictions: [],
          timestamp: timestamp,
          error: 'Processing failed'
        });
        allCounts.push(0);
      }
    }

    // Create output video from annotated frames
    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(path.join(annotatedFramesDir, 'frame_%04d.jpg'))
        .inputFPS(2)
        .videoCodec('libx264')
        .outputOptions([
          '-pix_fmt yuv420p',
          '-crf 23',
          '-preset medium'
        ])
        .output(outputVideoPath)
        .on('end', () => {
          console.log('Annotated video created successfully');
          resolve();
        })
        .on('error', (err) => {
          console.error('Error creating video:', err);
          reject(err);
        })
        .run();
    });

    // Calculate statistics
    const validFrames = frameResults.filter(f => !f.error);
    const totalPeople = validFrames.reduce((sum, frame) => sum + frame.count, 0);
    const avgPeople = validFrames.length > 0 ? (totalPeople / validFrames.length).toFixed(1) : 0;
    const maxPeople = validFrames.length > 0 ? Math.max(...validFrames.map(f => f.count)) : 0;

    res.json({
      success: true,
      originalVideo: `/uploads/${req.file.filename}`,
      annotatedVideo: `/uploads/${outputVideoName}`,
      totalFrames: frameResults.length,
      validFrames: validFrames.length,
      results: frameResults,
      statistics: {
        totalPeople: totalPeople,
        averagePeople: avgPeople,
        maxPeople: maxPeople,
        duration: (frameResults.length * 0.5).toFixed(1) + 's'
      }
    });

  } catch (err) {
    console.error('Video processing error:', err);
    res.status(500).json({ success: false, error: err.message || 'Video processing failed' });
  }
});

// -------------- Socket.IO real-time ----------------

const MAX_CONCURRENT = Number(process.env.MAX_CONCURRENT || 2);
let inFlight = 0;
const queue = [];

async function callRoboflowWithBase64(base64) {
  const body = {
    api_key: ROBOFLOW_API_KEY,
    inputs: { image: { type: 'base64', value: base64 } }
  };
  const resp = await axios.post(ROBOFLOW_ENDPOINT, body, { 
    headers: { 'Content-Type': 'application/json' }, 
    timeout: 30000 
  });
  return resp.data;
}

async function processFrameAndEmit(clientSocket, frameBase64) {
  if (inFlight >= MAX_CONCURRENT) {
    queue.push({ socket: clientSocket, frameBase64 });
    return;
  }
  inFlight++;
  try {
    const buf = Buffer.from(frameBase64, 'base64');
    const rf = await callRoboflowWithBase64(frameBase64);
    
    const output = rf.outputs?.[0] || {};
    const predictions = output.predictions?.predictions || [];
    const count = output.count_objects || predictions.length;
    
    // Check threshold for real-time alerts
    if (count >= ALERT_THRESHOLD && Date.now() - lastAlertTime > ALERT_COOLDOWN) {
      lastAlertTime = Date.now();
      console.log(`ALERT: Threshold exceeded (${count} people)`);
    }
    
    // Process images
    const processedImage = await drawDotsOnImage(buf, predictions);
    const annotatedImageBase64 = processedImage.toString('base64');
    const heatmapImage = await generateHeatmap(buf, predictions);
    const heatmapImageBase64 = heatmapImage.toString('base64');
    
    const payload = {
      success: true,
      count: count,
      annotatedImage: annotatedImageBase64,
      heatmapImage: heatmapImageBase64,
      predictions: predictions,
      threshold: ALERT_THRESHOLD
    };
    
    io.emit('prediction', payload);
  } catch (err) {
    console.error('Roboflow processing error:', err);
    io.emit('prediction', { 
      success: false, 
      error: err.message || 'Inference error' 
    });
  } finally {
    inFlight--;
    if (queue.length > 0 && inFlight < MAX_CONCURRENT) {
      const next = queue.shift();
      processFrameAndEmit(next.socket, next.frameBase64);
    }
  }
}

async function drawDotsOnImage(imageBuffer, predictions) {
  try {
    if (!Array.isArray(predictions)) {
      console.warn('Predictions is not an array, using empty array');
      predictions = [];
    }

    const image = await sharp(imageBuffer).toBuffer();
    const img = await loadImage(image);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
    predictions.forEach(pred => {
      try {
        const x = pred.x;
        const y = pred.y;
        const dotSize = Math.min(20, Math.max(5, Math.sqrt(pred.width * pred.height) / 5));
        
        if (x && y) {
          ctx.beginPath();
          ctx.arc(x, y, dotSize, 0, Math.PI * 2);
          ctx.fill();
          
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          ctx.fillText(`${Math.round(pred.confidence * 100)}%`, x + dotSize + 2, y);
          ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        }
      } catch (err) {
        console.warn('Error drawing prediction:', err);
      }
    });
    
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } catch (err) {
    console.error('Error in drawDotsOnImage:', err);
    throw err;
  }
}

async function generateHeatmap(imageBuffer, predictions) {
  try {
    if (!Array.isArray(predictions)) {
      predictions = [];
    }

    const image = await sharp(imageBuffer).toBuffer();
    const img = await loadImage(image);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    
    ctx.globalAlpha = 0.7;
    ctx.drawImage(img, 0, 0);
    ctx.globalAlpha = 1.0;
    
    predictions.forEach(pred => {
      if (pred.x && pred.y) {
        const radius = Math.min(img.width, img.height) * 0.1;
        const grd = ctx.createRadialGradient(
          pred.x, pred.y, 0, 
          pred.x, pred.y, radius
        );
        
        grd.addColorStop(0, 'rgba(255, 0, 0, 0.8)');
        grd.addColorStop(0.5, 'rgba(255, 255, 0, 0.4)');
        grd.addColorStop(1, 'rgba(0, 0, 255, 0)');
        
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(pred.x, pred.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } catch (err) {
    console.error('Error generating heatmap:', err);
    throw err;
  }
}

io.on('connection', (socket) => {
  socket.on('frame', (data) => {
    if (!data || !data.imageBase64) {
      return socket.emit('prediction', { success: false, error: 'No frame provided' });
    }

    if (queue.length > 60) {
      return socket.emit('prediction', { success: false, error: 'Server busy, frame dropped' });
    }
    socket.emit('ack', { received: true });
    processFrameAndEmit(socket, data.imageBase64);
  });

  socket.on('updateThreshold', (data) => {
    if (data && data.threshold) {
      ALERT_THRESHOLD = parseInt(data.threshold);
      console.log(`Threshold updated to ${ALERT_THRESHOLD}`);
      io.emit('thresholdUpdated', { threshold: ALERT_THRESHOLD });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected', socket.id);
  });
});

app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

server.listen(PORT, () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});

