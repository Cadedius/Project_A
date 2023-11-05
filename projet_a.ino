#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

// Threshold for significant movement in G (acceleration)
const float accelerationThreshold = 2.5;
const int numSamples = 119; // Number of samples for gesture detection

int samplesRead = numSamples;
bool gestureDetected = false;
bool yesDetected = false;
bool noDetected = false;
float yesProbability = 0.0; // Probability of "yes" gesture
int currentQuestion = 0; // Current question number
bool quizCompleted = false;
int score = 0; // Quiz score

// Quiz questions
const char* questions[] = {
  "Question 1: Est-ce que le ciel est bleu ?",
  "Question 2: Lyon est la capitale de la France ?",
  "Question 3: Il fait chaud en été ?"
};

// Global variables for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// Resolver for all TFLM operations (you can remove unnecessary operations to reduce code size)
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM (adjust size based on your model)
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Mapping of gesture index to name
const char* GESTURES[] = {
  "yes",
  "no"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Échec de l'initialisation de l'IMU !");
    while (1);
  }

  // Display accelerometer and gyroscope sampling rates
  Serial.print("Taux d'échantillonnage de l'accéléromètre = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Taux d'échantillonnage du gyroscope = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // Get the TFL model representation from the model array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Incompatibilité du schéma du modèle !");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for input and output tensors of the model
  tflInterpreter->AllocateTensors();

  // Get pointers to the input and output tensors of the model
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Wait for significant movement for each new question
  while (!quizCompleted && samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // Read acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // Calculate the sum of absolute values
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // Check if it exceeds the threshold
      if (aSum >= accelerationThreshold) {
        // Reset sample counter and gesture detection
        samplesRead = 0;
        gestureDetected = false;
        yesDetected = false;
        noDetected = false;

        // Display the question
        Serial.println(questions[currentQuestion]);
        break;
      }
    }
  }

  // Check if all required samples have been read since the last significant movement
  while (!quizCompleted && samplesRead < numSamples && !gestureDetected) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // Read acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Normalize IMU data between 0 and 1 and store it in the model's input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Perform inference
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Échec de l'inférence !");
          while (1);
          return;
        }

        // Loop through the values in the model's output tensor
        for (int i = 0; i < NUM_GESTURES; i++) {
          if (tflOutputTensor->data.f[i] > 0.9) {
            gestureDetected = true;
            yesProbability = tflOutputTensor->data.f[i];
            if (i == 0) {
              yesDetected = true;
            } else if (i == 1) {
              noDetected = true;
            }
          }
        }

        // Display the response based on the detected gesture
        if (yesDetected) {
          if (yesProbability >= 0.8) {
            Serial.println("Réponse : Oui (✓).");
          } else {
            Serial.println("Réponse : Oui, mais la probabilité est faible.");
          }
        } else if (noDetected) {
          Serial.println("Réponse : Non (X).");
        }

        // Move to the next question or finish the quiz
        currentQuestion++;
        if (currentQuestion == 1) {
          if (yesDetected == true) {
            score++;
          } else {
            score--;
          }
        }
        if (currentQuestion == 2) {
          if (noDetected == true) {
            score++;
          } else {
            score--;
          }
        }
        if (currentQuestion >= 3) {
          if (yesDetected == true) {
            score++;
          } else {
            score--;
          }
          quizCompleted = true;
          Serial.println("Quiz terminé !");
          Serial.print("Votre score est de: ");
          Serial.println(score);
        }
      }
    }
  }
}
