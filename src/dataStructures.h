
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct PerfEval { // performance evaluation for different combination
    std::string detectorType;           // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    std::string descriptorType;         // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    std::string matcherType;            // MAT_BF, MAT_FLANN
    std::string selectorType;           // SEL_NN, SEL_KNN
    int numKeyPointsPerFrame[10];       // Count the number of keypoints for all 10 images
    int numKeyPointsPerROI[10];         // Count the number of keypoints on the preceding vehicle for all 10 images 
    int numKeyPointsMatched[10];        // Count the number of keypoints matched for all 10 images
    double timeKeyPointsDetection[10];  // Log the time it takes for keypoint detection
    double timeDescriptorExtraction[10];// Log the time it takes for descriptor extraction
    double timeMatching[10];            // Log the time it takes for matching
};

struct ReturnVal {
    int numPoints;
    double deltaTime;
};

#endif /* dataStructures_h */
