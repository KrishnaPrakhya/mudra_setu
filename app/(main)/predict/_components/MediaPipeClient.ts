import {
  HolisticLandmarker,
  FilesetResolver,
  HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Define the structure of our landmark data for clarity
export interface LandmarkData {
  results: HolisticLandmarkerResult;
  keypoints: number[];
}

// Indices now match the focused set used in the Python model training.
const UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16];
const POSE_LANDMARK_COUNT = 7 * 4; // 7 landmarks, 4 values (x,y,z,visibility)
const HAND_LANDMARK_COUNT = 21 * 3; // 21 landmarks, 3 values (x,y,z)

class MediaPipeClient {
  private holisticLandmarker: HolisticLandmarker | null = null;

  async initialize(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
    );
    this.holisticLandmarker = await HolisticLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          // **FIX**: Corrected the model asset path to the official holistic landmarker bundle.
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
      }
    );
  }

  detect(video: HTMLVideoElement): LandmarkData | null {
    if (!this.holisticLandmarker) return null;

    // **IMPROVEMENT**: Added a try-catch block for robust error handling during detection.
    try {
      const results = this.holisticLandmarker.detectForVideo(
        video,
        performance.now()
      );
      const keypoints = this.extractKeypoints(results);
      return { results, keypoints };
    } catch (error) {
      console.error("Error during landmark detection:", error);
      return null;
    }
  }

  private extractKeypoints(results: HolisticLandmarkerResult): number[] {
    const pose = new Array(POSE_LANDMARK_COUNT).fill(0);
    // **IMPROVEMENT**: Logic now explicitly extracts only the 7 required upper body landmarks.
    if (results.poseLandmarks && results.poseLandmarks[0]) {
      UPPER_BODY_INDICES.forEach((idx, i) => {
        const lm = results.poseLandmarks[0][idx];
        if (lm) {
          pose[i * 4] = lm.x;
          pose[i * 4 + 1] = lm.y;
          pose[i * 4 + 2] = lm.z;
          pose[i * 4 + 3] = lm.visibility ?? 0;
        }
      });
    }

    const lh = new Array(HAND_LANDMARK_COUNT).fill(0);
    if (results.leftHandLandmarks && results.leftHandLandmarks[0]) {
      results.leftHandLandmarks[0].forEach((lm, i) => {
        lh[i * 3] = lm.x;
        lh[i * 3 + 1] = lm.y;
        lh[i * 3 + 2] = lm.z;
      });
    }

    const rh = new Array(HAND_LANDMARK_COUNT).fill(0);
    if (results.rightHandLandmarks && results.rightHandLandmarks[0]) {
      results.rightHandLandmarks[0].forEach((lm, i) => {
        rh[i * 3] = lm.x;
        rh[i * 3 + 1] = lm.y;
        rh[i * 3 + 2] = lm.z;
      });
    }

    // This creates an array with 154 values, matching the backend's expectation.
    return [...pose, ...lh, ...rh];
  }
}

export const mediaPipeClient = new MediaPipeClient();