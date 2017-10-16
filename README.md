# Face Sketch Recognition

Based on [The FaceSketchID System: Matching Facial Composites to Mugshots](http://ieeexplore.ieee.org/document/6912947/ "IEEE"),
the project tries to address the performance and accuracy issues in matching images to face sketches.

Note that this README is a work-in-progress. Please bear any typos,
lack of documentation and bad grammar.

## Prerequisites

+ Python 3.5
+ OpenCV 3.3
+ STASM 4.1
+ CUFS Database


## Setting up CUFS Database

Download CUFS database fromhttp://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

Make sure the database is extracted into the following structure of directories.

```
$PROJECT_HOME
├── data
│   ├── photo_data_points
│   ├── photos
│   ├── photos_cropped
│   ├── sketch_data_points
│   ├── sketches
│   └── sketches_cropped
```