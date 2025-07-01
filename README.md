# Mudra Setu - Real-Time Sign Language Recognition Platform


Mudra Setu is a cutting-edge real-time sign language recognition platform that bridges the communication gap between the deaf/hard-of-hearing community and others using advanced AI technology.

## 🌟 Key Features

- **Real-Time Recognition**: Instant sign language interpretation through your webcam
- **High Accuracy**: Utilizing advanced attention-based neural networks with 98.80% accuracy
- **Hand Landmark Visualization**: Real-time visual feedback with detailed hand tracking
- **Responsive Design**: Beautiful, modern UI that works across devices
- **Performance Optimized**: Maintains high FPS for smooth recognition
- **User-Friendly Interface**: Simple one-click start/stop capture system

## 🚀 Technical Highlights

- **Frontend**: Next.js 14 with TypeScript for type-safe, modern web development
- **AI Model**: Custom attention-based neural network for sign language recognition
- **Real-Time Processing**: Efficient frame capture and processing pipeline
- **Hand Tracking**: Advanced hand landmark detection and visualization
- **Backend**: Fast API server with Python for ML model inference

## Getting Started

First, install the dependencies and run the development server:

### Frontend Setup

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

### Backend Setup

```bash
# Navigate to server directory
cd server

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python app.py
```

Open [http://localhost:3000](http://localhost:3000) with your browser to access the application.

## 🏗️ Project Structure

```
├── app/                    # Next.js application routes
│   ├── (main)/            # Main application pages
│   │   ├── predict/       # Real-time prediction interface
│   │   ├── visualize/     # Data visualization components
│   │   └── model/        # Model performance metrics
│   └── api/              # API routes
├── model/                # ML model files and weights
├── server/              # Python FastAPI backend
└── components/         # Reusable React components
```

## 🤖 ML Model Architecture

Our sign language recognition model uses an attention-based neural network architecture that achieves 98.80% accuracy. Key features:

- Attention mechanism for focusing on important hand landmarks
- Enhanced feature extraction for subtle hand movements
- Robust scaling and preprocessing pipeline
- Trained on a comprehensive sign language dataset

## 💡 How It Works

1. **Capture**: Real-time video capture through the browser
2. **Processing**: Frame extraction and preprocessing
3. **Detection**: Hand landmark detection and tracking
4. **Recognition**: ML model inference for sign recognition
5. **Visualization**: Real-time display of results and hand tracking

## 🔧 Advanced Configuration

The application can be configured through various environment variables:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000  # FastAPI backend URL
MODEL_PATH=./model/sign_model_focused_enhanced_attention_v2_0.9880_prior1  # Model path
```

## 🚀 Deployment

### Frontend Deployment

The frontend can be easily deployed on Vercel:

```bash
vercel deploy
```

### Backend Deployment

The FastAPI backend can be deployed using Docker:

```bash
cd server
docker build -t mudra-setu-backend .
docker run -p 8000:8000 mudra-setu-backend
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - FastAPI automatic documentation
- [Model Architecture](./model/README.md) - Detailed ML model documentation
- [Contributing Guidelines](./CONTRIBUTING.md) - How to contribute to the project

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

- Built with Next.js, FastAPI, and TensorFlow
