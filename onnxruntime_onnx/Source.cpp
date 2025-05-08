#include <iostream>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <stdexcept>
#include "httplib.h"

using namespace std;
using json = nlohmann::json;

class UserBehaviorPredictor {
private:
    vector<double> scaleMin;
    vector<double> scaleRange;
    vector<string> featureNames;
    vector<string> numericalFeatures;
    vector<string> categoricalFeatures;
    vector<string> targetClasses;
    string modelPath;
    int inputSize = 0;
    int numClasses = 0;
    bool useRealONNX = false;

public:
    UserBehaviorPredictor(const string& modelPath, const string& scalerPath, const string& metadataPath)
        : modelPath(modelPath) {

        loadMetadata(metadataPath);
        loadScalerParams(scalerPath);

        cout << "Predictor initialized (Mock mode)" << endl;
    }

    void loadMetadata(const string& metadataPath) {
        ifstream file(metadataPath);
        if (!file.is_open()) {
            throw runtime_error("Cannot open metadata file: " + metadataPath);
        }

        json j;
        file >> j;

        if (j.contains("model_info")) {
            inputSize = j["model_info"]["input_size"];
            numClasses = j["model_info"]["num_classes"];
            targetClasses = j["model_info"]["target_classes"].get<vector<string>>();
        }

        if (j.contains("preprocessing")) {
            if (j["preprocessing"].contains("numerical_features")) {
                numericalFeatures = j["preprocessing"]["numerical_features"].get<vector<string>>();
            }
            if (j["preprocessing"].contains("categorical_features")) {
                categoricalFeatures = j["preprocessing"]["categorical_features"].get<vector<string>>();
            }
        }

        featureNames.clear();
        featureNames.insert(featureNames.end(), numericalFeatures.begin(), numericalFeatures.end());
        featureNames.insert(featureNames.end(), categoricalFeatures.begin(), categoricalFeatures.end());
    }

    void loadScalerParams(const string& scalerPath) {
        ifstream file(scalerPath);
        if (!file.is_open()) {
            throw runtime_error("Cannot open scaler file: " + scalerPath);
        }

        json j;
        file >> j;

        scaleMin = j["mean"].get<vector<double>>();
        scaleRange = j["scale"].get<vector<double>>();

        if (scaleMin.size() != numericalFeatures.size()) {
            throw runtime_error("Scaler parameters size doesn't match numerical features");
        }
    }

    vector<float> preprocessInput(const vector<double>& input) {
        if (input.size() != inputSize) {
            throw runtime_error("Input data size mismatch. Expected: " + to_string(inputSize) +
                ", Got: " + to_string(input.size()));
        }

        vector<float> scaled(input.size());

        // Scale only numerical features
        for (size_t i = 0; i < numericalFeatures.size(); i++) {
            scaled[i] = static_cast<float>((input[i] - scaleMin[i]) / scaleRange[i]);
        }

        // Don't scale categorical features
        for (size_t i = numericalFeatures.size(); i < input.size(); i++) {
            scaled[i] = static_cast<float>(input[i]);
        }

        return scaled;
    }

    vector<float> predict(const vector<double>& inputData) {
        auto scaledInput = preprocessInput(inputData);

        // Mock prediction logic
        float addtocart_score = 0.0f;
        float transaction_score = 0.0f;
        float view_score = 0.0f;

        // Process numerical features
        for (size_t i = 0; i < numericalFeatures.size(); i++) {
            float val = scaledInput[i];

            if (numericalFeatures[i] == "cart_abandonment") {
                addtocart_score += val * 1.5f;
                transaction_score -= val * 1.2f;
            }
            else if (numericalFeatures[i] == "session_transaction") {
                transaction_score += val * 1.8f;
            }
            else if (numericalFeatures[i] == "session_addtocart") {
                addtocart_score += val * 1.3f;
            }
            else if (numericalFeatures[i] == "session_view") {
                view_score += val * 1.1f;
            }
            else {
                addtocart_score += val * 0.4f;
                transaction_score += val * 0.3f;
                view_score += val * 0.3f;
            }
        }

        // Process categorical features
        for (size_t i = numericalFeatures.size(); i < scaledInput.size(); i++) {
            float val = scaledInput[i];
            int catIndex = i - numericalFeatures.size();

            if (catIndex == 0 && val == 0) { // prev_event == "addtocart"
                addtocart_score += 1.0f;
            }
            else if (catIndex == 0 && val == 1) { // prev_event == "transaction"
                transaction_score += 1.2f;
            }
            else if (catIndex == 3 && val > 3) { // time_diff_category very long
                view_score += 0.5f;
            }
        }

        // Apply softmax to get probabilities
        float total = exp(addtocart_score) + exp(transaction_score) + exp(view_score);

        vector<float> output = {
            exp(addtocart_score) / total,
            exp(transaction_score) / total,
            exp(view_score) / total
        };

        return output;
    }

    json predictAndFormat(const vector<double>& inputData) {
        auto predictions = predict(inputData);

        // Find the action with highest probability
        int maxIdx = 0;
        for (int i = 1; i < predictions.size(); i++) {
            if (predictions[i] > predictions[maxIdx]) {
                maxIdx = i;
            }
        }

        // Create response JSON
        json response;
        response["predicted_action"] = targetClasses[maxIdx];

        // Add all probabilities
        json probabilities;
        for (size_t i = 0; i < targetClasses.size(); i++) {
            probabilities[targetClasses[i]] = predictions[i];
        }
        response["probabilities"] = probabilities;

        return response;
    }
};

// RESTful API Server
class UserBehaviorAPIServer {
private:
    UserBehaviorPredictor predictor;
    httplib::Server server;
    int port;

public:
    UserBehaviorAPIServer(int port = 8080)
        : predictor("user_behavior_model.onnx", "scaler_params.json", "model_metadata.json"),
        port(port) {

        setupRoutes();
    }

    void setupRoutes() {
        // POST /predict_behavior endpoint
        server.Post("/predict_behavior", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse JSON from request body
                json requestData = json::parse(req.body);

                // Extract time series data
                vector<double> inputData;
                if (requestData.contains("data")) {
                    inputData = requestData["data"].get<vector<double>>();
                }
                else {
                    res.status = 400;
                    res.set_content("{\"error\": \"Missing 'data' field in request\"}", "application/json");
                    return;
                }

                // Make prediction
                json responseData = predictor.predictAndFormat(inputData);

                // Set response
                res.set_content(responseData.dump(), "application/json");
                res.status = 200;

            }
            catch (const exception& e) {
                res.status = 500;
                json errorResponse;
                errorResponse["error"] = e.what();
                res.set_content(errorResponse.dump(), "application/json");
            }
            });

        // GET /health endpoint
        server.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            json health;
            health["status"] = "healthy";
            health["service"] = "User Behavior Prediction API";
            health["mode"] = "mock";
            res.set_content(health.dump(), "application/json");
            });

        // GET /model_info endpoint
        server.Get("/model_info", [this](const httplib::Request& req, httplib::Response& res) {
            json info;
            info["input_size"] = 25;
            info["output_classes"] = { "addtocart", "transaction", "view" };
            info["mode"] = "mock";
            info["onnx_runtime"] = "not linked";
            res.set_content(info.dump(), "application/json");
            });
    }

    void start() {
        cout << "Starting User Behavior Prediction API on port " << port << endl;
        cout << "Endpoints:" << endl;
        cout << "  POST /predict_behavior" << endl;
        cout << "  GET  /health" << endl;
        cout << "  GET  /model_info" << endl;
        cout << endl;

        server.listen("0.0.0.0", port);
    }
};

int main() {
    try {
        cout << "=== User Behavior Prediction API Server ===" << endl;

        // Start API server
        UserBehaviorAPIServer apiServer(8080);
        apiServer.start();

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}