# Deploy Core

The `DeployCore` module defines the abstract functionalities for all components, including core inference capabilities, 2D detection features, SAM functionalities, and more. Beyond merely defining abstract functions, DeployCore also provides external encapsulations for certain algorithms. When implementing an algorithm, developers only need to focus on completing the key processes outlined in these definitions to achieve seamless algorithm deployment.

## Functionality

`DeployCore` is designed to provide abstract interface definitions for the functionalities of all modules, as well as abstract base classes containing reusable code.

- Abstract core inference functionality: `BaseInferCore`  
- Abstract 2D detection functionality: `BaseDetection2DModel`  
- Abstract SAM functionality: `BaseSamModel`  
- Plug-and-play asynchronous pipeline base class: `BaseAsyncPipeline`

## Structure

The entire project code is divided into three parts:  
  1. Abstract interface classes for functional modules  
  2. Abstract base classes for certain functional modules  
  3. Base classes for the asynchronous inference pipeline framework

code structure:
  ```bash
  deploy_core
  |-- CMakeLists.txt
  |-- README.md
  |-- include
  |   `-- deploy_core
  |       |-- base_infer_core.h
  |       |-- base_detection.h
  |       |-- base_sam.h
  |       |-- async_pipeline.h
  |       |-- async_pipeline_impl.h
  |       |-- block_queue.h
  |       |-- common_defination.h
  |       `-- wrapper.h
  `-- src
      |-- base_detection.cpp
      |-- base_infer_core.cpp
      `-- base_sam.cpp
  ```


  - Abstract interface classes for functional modules
    ```bash
    |-- base_infer_core.h
    |-- base_detection.h
    |-- base_sam.h
    ```
    1. **`base_infer_core.h`**: Defines the core inference functionalities and related abstract classes, while also providing an abstract base class for the foundational features of the inference core module.  
    2. **`base_detection.h`**: Defines the abstract base class for 2D detection functionalities.  
    3. **`base_sam.h`**: Defines the abstract base class for SAM functionalities. 

  - Base classes for the asynchronous inference pipeline framework
    ```bash
    |-- async_pipeline.h
    |-- async_pipeline_impl.h
    |-- block_queue.h
    |-- common_defination.h
    `-- wrapper.h
    ```
    1. **`async_pipeline.h`** and **`async_pipeline_impl.h`**: Define the asynchronous inference framework and its implementation.  
    2. **`block_queue.h`**: Implements the blocking queue.  
    3. **`common_defination.h`**: Contains common definitions, such as 2D bounding boxes.  
    4. **`wrapper.h`**: Provides wrappers for certain classes, such as the encapsulation of OpenCV's `cv::Mat` format.
  

## TODO

