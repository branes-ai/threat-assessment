#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Forward declarations
class Camera;
class FrameProcessor;
class ObjectDetector;
class ThreatAssessor;
class SystemController;

// Represents a detected object with position and metadata
struct DetectedObject {
    int id;
    std::string className;
    cv::Rect boundingBox;
    float confidence;
    std::vector<float> features;
};

// Represents the threat assessment result
struct ThreatAssessment {
    int objectId;
    float threatScore;  // 0-1 where 1 is highest threat
    std::string threatCategory;
    std::string recommendedAction;
};

// Thread-safe queue for passing frames between components
template<typename T>
class SafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cond_.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

// Camera interface for capturing frames
class Camera {
private:
    int cameraId_;
    cv::VideoCapture cap_;
    bool isRunning_;
    std::thread captureThread_;
    
public:
    SafeQueue<cv::Mat> frameQueue;
    
    Camera(int id, const std::string& streamUrl) : cameraId_(id), isRunning_(false) {
        cap_.open(streamUrl);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Failed to open camera " + std::to_string(id));
        }
    }
    
    void start() {
        isRunning_ = true;
        captureThread_ = std::thread(&Camera::captureLoop, this);
    }
    
    void stop() {
        isRunning_ = false;
        if (captureThread_.joinable()) {
            captureThread_.join();
        }
    }
    
private:
    void captureLoop() {
        while (isRunning_) {
            cv::Mat frame;
            if (cap_.read(frame)) {
                frameQueue.push(frame.clone());
            } else {
                std::cerr << "Warning: Failed to read frame from camera " << cameraId_ << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
};

// Processes frames from multiple cameras
class FrameProcessor {
private:
    std::vector<Camera*> cameras_;
    SafeQueue<std::pair<int, cv::Mat>> processedFrames_;
    ObjectDetector* detector_;
    bool isRunning_;
    std::thread processingThread_;
    
public:
    FrameProcessor(ObjectDetector* detector) : detector_(detector), isRunning_(false) {}
    
    void addCamera(Camera* camera) {
        cameras_.push_back(camera);
    }
    
    void start() {
        isRunning_ = true;
        processingThread_ = std::thread(&FrameProcessor::processLoop, this);
    }
    
    void stop() {
        isRunning_ = false;
        if (processingThread_.joinable()) {
            processingThread_.join();
        }
    }
    
    SafeQueue<std::pair<int, cv::Mat>>& getProcessedFrames() {
        return processedFrames_;
    }
    
private:
    void processLoop() {
        while (isRunning_) {
            for (size_t i = 0; i < cameras_.size(); i++) {
                if (!cameras_[i]->frameQueue.empty()) {
                    cv::Mat frame = cameras_[i]->frameQueue.pop();
                    // Pre-process frame (resize, normalize, etc.)
                    cv::Mat processedFrame;
                    cv::resize(frame, processedFrame, cv::Size(640, 480));
                    processedFrames_.push({static_cast<int>(i), processedFrame});
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
};

// Object detection using a deep learning model
class ObjectDetector {
private:
    torch::jit::script::Module model_;
    std::vector<std::string> classNames_;
    float confidenceThreshold_;
    
public:
    ObjectDetector(const std::string& modelPath, const std::string& classNamesFile, float confidenceThreshold = 0.5) 
        : confidenceThreshold_(confidenceThreshold) {
        try {
            // Load the PyTorch model
            model_ = torch::jit::load(modelPath);
            model_.eval();
            
            // Load class names
            std::ifstream file(classNamesFile);
            std::string line;
            while (std::getline(file, line)) {
                classNames_.push_back(line);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    std::vector<DetectedObject> detectObjects(const cv::Mat& frame) {
        std::vector<DetectedObject> detectedObjects;
        
        try {
            // Convert OpenCV Mat to torch tensor
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(640, 640));
            cv::Mat converted;
            resized.convertTo(converted, CV_32F, 1.0/255.0);
            
            // HWC to CHW format
            cv::Mat channels[3];
            cv::split(converted, channels);
            
            auto tensorR = torch::from_blob(channels[0].data, {640, 640}, torch::kFloat32);
            auto tensorG = torch::from_blob(channels[1].data, {640, 640}, torch::kFloat32);
            auto tensorB = torch::from_blob(channels[2].data, {640, 640}, torch::kFloat32);
            
            auto inputTensor = torch::cat({tensorR.unsqueeze(0), 
                                          tensorG.unsqueeze(0), 
                                          tensorB.unsqueeze(0)}, 0)
                                         .unsqueeze(0);
            
            // Normalize tensor
            inputTensor = inputTensor.sub_(0.5).div_(0.5);
            
            // Run inference
            torch::NoGradGuard no_grad;
            auto output = model_.forward({inputTensor}).toTuple();
            auto detections = output->elements()[0].toTensor();
            
            // Process detections
            for (int i = 0; i < detections.size(1); i++) {
                float confidence = detections[0][i][4].item<float>();
                
                if (confidence >= confidenceThreshold_) {
                    float x = detections[0][i][0].item<float>();
                    float y = detections[0][i][1].item<float>();
                    float w = detections[0][i][2].item<float>();
                    float h = detections[0][i][3].item<float>();
                    int classId = detections[0][i][5].item<int>();
                    
                    DetectedObject obj;
                    obj.id = i;
                    obj.className = classNames_[classId];
                    obj.boundingBox = cv::Rect(x - w/2, y - h/2, w, h);
                    obj.confidence = confidence;
                    
                    // Extract features for later use in threat assessment
                    std::vector<float> features;
                    for (int j = 6; j < 6 + 128; j++) {  // Assuming 128 feature dimensions
                        if (j < detections.size(2)) {
                            features.push_back(detections[0][i][j].item<float>());
                        }
                    }
                    obj.features = features;
                    
                    detectedObjects.push_back(obj);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during object detection: " << e.what() << std::endl;
        }
        
        return detectedObjects;
    }
};

// Threat assessment model
class ThreatAssessor {
private:
    torch::jit::script::Module model_;
    std::vector<std::string> threatCategories_;
    
public:
    ThreatAssessor(const std::string& modelPath, const std::string& categoriesFile) {
        try {
            // Load the PyTorch model
            model_ = torch::jit::load(modelPath);
            model_.eval();
            
            // Load threat categories
            std::ifstream file(categoriesFile);
            std::string line;
            while (std::getline(file, line)) {
                threatCategories_.push_back(line);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading threat assessment model: " << e.what() << std::endl;
            throw;
        }
    }
    
    ThreatAssessment assessThreat(const DetectedObject& object) {
        ThreatAssessment assessment;
        assessment.objectId = object.id;
        
        try {
            // Convert object features to tensor
            auto featuresTensor = torch::from_blob(object.features.data(), 
                                                 {1, static_cast<long>(object.features.size())}, 
                                                 torch::kFloat32);
            
            // Add object class as one-hot encoding
            auto classVector = torch::zeros({1, static_cast<long>(classNames_.size())});
            for (size_t i = 0; i < classNames_.size(); i++) {
                if (classNames_[i] == object.className) {
                    classVector[0][i] = 1.0;
                    break;
                }
            }
            
            // Combine features
            auto inputTensor = torch::cat({featuresTensor, classVector}, 1);
            
            // Run inference
            torch::NoGradGuard no_grad;
            auto output = model_.forward({inputTensor}).toTuple();
            auto threatScore = output->elements()[0].toTensor().item<float>();
            auto categoryIndex = output->elements()[1].toTensor().item<int>();
            
            assessment.threatScore = threatScore;
            assessment.threatCategory = threatCategories_[categoryIndex];
            
            // Generate recommended action based on threat score
            if (threatScore > 0.8) {
                assessment.recommendedAction = "Immediate response required";
            } else if (threatScore > 0.5) {
                assessment.recommendedAction = "Alert security personnel";
            } else if (threatScore > 0.2) {
                assessment.recommendedAction = "Monitor closely";
            } else {
                assessment.recommendedAction = "No action needed";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during threat assessment: " << e.what() << std::endl;
        }
        
        return assessment;
    }
    
private:
    std::vector<std::string> classNames_;  // Need to initialize this from detector
};

// Main system controller
class SystemController {
private:
    std::vector<Camera*> cameras_;
    FrameProcessor* processor_;
    ObjectDetector* detector_;
    ThreatAssessor* assessor_;
    bool isRunning_;
    std::thread mainThread_;
    
public:
    SystemController() : isRunning_(false) {
        // Initialize components
        detector_ = new ObjectDetector("models/yolov5.pt", "models/coco.names");
        assessor_ = new ThreatAssessor("models/threat_model.pt", "models/threat_categories.txt");
        processor_ = new FrameProcessor(detector_);
    }
    
    ~SystemController() {
        stop();
        delete processor_;
        delete detector_;
        delete assessor_;
        for (auto camera : cameras_) {
            delete camera;
        }
    }
    
    void addCamera(const std::string& streamUrl) {
        int cameraId = cameras_.size();
        Camera* camera = new Camera(cameraId, streamUrl);
        cameras_.push_back(camera);
        processor_->addCamera(camera);
    }
    
    void start() {
        // Start all cameras
        for (auto camera : cameras_) {
            camera->start();
        }
        
        // Start frame processor
        processor_->start();
        
        // Start main processing loop
        isRunning_ = true;
        mainThread_ = std::thread(&SystemController::mainLoop, this);
        
        std::cout << "System started with " << cameras_.size() << " cameras" << std::endl;
    }
    
    void stop() {
        isRunning_ = false;
        
        if (mainThread_.joinable()) {
            mainThread_.join();
        }
        
        processor_->stop();
        
        for (auto camera : cameras_) {
            camera->stop();
        }
        
        std::cout << "System stopped" << std::endl;
    }
    
private:
    void mainLoop() {
        auto& processedFrames = processor_->getProcessedFrames();
        
        while (isRunning_) {
            if (!processedFrames.empty()) {
                auto [cameraId, frame] = processedFrames.pop();
                
                // Detect objects
                std::vector<DetectedObject> objects = detector_->detectObjects(frame);
                
                // Assess threats for each object
                std::vector<ThreatAssessment> assessments;
                for (const auto& obj : objects) {
                    ThreatAssessment assessment = assessor_->assessThreat(obj);
                    assessments.push_back(assessment);
                    
                    // Log high threats
                    if (assessment.threatScore > 0.5) {
                        std::cout << "HIGH THREAT DETECTED: Camera " << cameraId 
                                  << ", Object " << obj.className 
                                  << ", Threat score: " << assessment.threatScore
                                  << ", Category: " << assessment.threatCategory 
                                  << ", Action: " << assessment.recommendedAction << std::endl;
                    }
                }
                
                // Visualize results (optional)
                visualizeResults(frame, objects, assessments);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void visualizeResults(cv::Mat& frame, 
                         const std::vector<DetectedObject>& objects,
                         const std::vector<ThreatAssessment>& assessments) {
        for (size_t i = 0; i < objects.size(); i++) {
            const auto& obj = objects[i];
            const auto& assessment = assessments[i];
            
            // Draw bounding box - color based on threat level
            cv::Scalar color;
            if (assessment.threatScore > 0.7) {
                color = cv::Scalar(0, 0, 255);  // Red for high threat
            } else if (assessment.threatScore > 0.4) {
                color = cv::Scalar(0, 165, 255);  // Orange for medium threat
            } else {
                color = cv::Scalar(0, 255, 0);  // Green for low threat
            }
            
            cv::rectangle(frame, obj.boundingBox, color, 2);
            
            // Add text
            std::string text = obj.className + " " + 
                              std::to_string(static_cast<int>(obj.confidence * 100)) + "% - Threat: " +
                              std::to_string(static_cast<int>(assessment.threatScore * 100)) + "%";
            
            cv::putText(frame, text, 
                       cv::Point(obj.boundingBox.x, obj.boundingBox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        
        // Display frame (in production, might want to stream this elsewhere)
        cv::imshow("Threat Detection", frame);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv) {
    try {
        SystemController system;
        
        // Add cameras - in production these would be loaded from config
        system.addCamera("rtsp://camera1_ip_address/stream");
        system.addCamera("rtsp://camera2_ip_address/stream");
        // For testing with webcam:
        // system.addCamera("0");  // Default camera
        
        system.start();
        
        std::cout << "Press Enter to stop the system..." << std::endl;
        std::cin.get();
        
        system.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
