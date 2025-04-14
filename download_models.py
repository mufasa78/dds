import os
import urllib.request
import sys
import time

def download_file(url, save_path):
    """
    Download a file from a URL and show progress
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the file to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        """Report download progress"""
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            s = "\r%5.1f%% %*d / %d" % (percent, len(str(total_size)), read_so_far, total_size)
            sys.stdout.write(s)
            if read_so_far >= total_size:
                sys.stdout.write("\n")
        else:
            sys.stdout.write("read %d\n" % (read_so_far,))
    
    print(f"Downloading {url} to {save_path}...")
    urllib.request.urlretrieve(url, save_path, report_progress)
    print(f"Downloaded {save_path} successfully!")

def main():
    """Download all required model files"""
    print("Downloading required model files for deepfake detection...")
    
    # Create models directory
    os.makedirs("models/weights", exist_ok=True)
    
    # Download face detection model
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    prototxt_path = "models/weights/deploy.prototxt"
    
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    caffemodel_path = "models/weights/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Download deepfake detection model
    model_url = "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_999_DeepFakeClassifier_EfficientNetB4_custom_face.pt"
    model_path = "models/weights/deepfake_model.pt"
    
    # Download each file
    if not os.path.exists(prototxt_path):
        download_file(prototxt_url, prototxt_path)
    else:
        print(f"{prototxt_path} already exists, skipping download.")
    
    if not os.path.exists(caffemodel_path):
        download_file(caffemodel_url, caffemodel_path)
    else:
        print(f"{caffemodel_path} already exists, skipping download.")
    
    if not os.path.exists(model_path):
        download_file(model_url, model_path)
    else:
        print(f"{model_path} already exists, skipping download.")
    
    print("\nAll model files downloaded successfully!")
    print("\nYou can now run the deepfake detection system with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
