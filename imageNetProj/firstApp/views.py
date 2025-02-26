from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt


class CompressedCNN(nn.Module):
    def __init__(self, num_classes):
        super(CompressedCNN, self).__init__()
        
        # Initial convolution block with fewer filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Reduced from 64 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Deep feature extraction blocks with reduced filters
        self.features = nn.Sequential(
            self._make_block(32, 64),    # Reduced from 64->128 to 32->64
            self._make_block(64, 128),   # Reduced from 128->256 to 64->128
            self._make_block(128, 256),  # Reduced from 256->512 to 128->256
            self._make_block(256, 256)   # Reduced from 512->512 to 256->256
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with reduced neurons
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # Reduced from 512->1024 to 256->512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Reduced from 1024->512 to 512->256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Reduced from 512->num_classes to 256->num_classes
        )
    
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize model and settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(settings.BASE_DIR, "models", "90%TrainedModel_compressed.pth")

# Ensure model directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Create model instance
model = CompressedCNN(num_classes=2)

# Load the model
if os.path.exists(model_path):
    try:
        # Create a quantized version of the model as in your training code
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        # Load the state dict directly - since your training code saves just the state_dict
        state_dict = torch.load(model_path, map_location=device)
        quantized_model.load_state_dict(state_dict)
        
        # Use the quantized model for inference
        model = quantized_model
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Warning: Model file not found at {model_path}")

class_labels = {0: "Laptop", 1: "Smartphone"}  # Swapped the labels

# Match the preprocessing exactly with training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def index(request):
    return render(request, 'index.html', {'title': 'Image Classifier'})


def predictImage(request):
    context = {'title': 'Image Classifier'}
    
    if request.method == 'POST' and request.FILES.getlist('filePath'):
        try:
            results = []
            files = request.FILES.getlist('filePath')
            fs = FileSystemStorage()
            
            for file_obj in files:
                # Validate file type
                if not file_obj.name.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
                    context['error'] = 'Please upload only image files (PNG, JPG, JPEG, WEBP)'
                    continue
                
                # Save file
                filename = fs.save(file_obj.name, file_obj)
                file_url = fs.url(filename)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                
                # Process image and predict
                img = Image.open(file_path).convert("RGB")
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_class = torch.argmax(probabilities)
                    predicted_label = class_labels[predicted_class.item()]
                    confidence = probabilities[predicted_class].item() * 100
                    
                    results.append({
                        'filePathName': file_url,
                        'predictedLabel': predicted_label,
                        'confidence': f'{confidence:.2f}%'
                    })
            
            context.update({
                'results': results,
                'success': True
            })
                
        except Exception as e:
            context['error'] = f'Error processing image: {str(e)}'
    
    return render(request, 'index.html', context)


@csrf_exempt
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    if not request.FILES:
        return JsonResponse({'error': 'No files uploaded'}, status=400)
    
    results = []
    try:
        fs = FileSystemStorage()
        
        # Iterate through all uploaded files
        for file_key in request.FILES:
            file_obj = request.FILES[file_key]
            
            # Validate file type
            if not file_obj.name.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
                results.append({
                    'filename': file_obj.name,
                    'error': 'Invalid file type'
                })
                continue
            
            try:
                # Save file
                filename = fs.save(file_obj.name, file_obj)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                
                # Process image
                img = Image.open(file_path).convert("RGB")
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_class = torch.argmax(probabilities)
                    predicted_label = class_labels[predicted_class.item()]
                    confidence = probabilities[predicted_class].item() * 100
                    
                    # Add prediction result
                    results.append({
                        'filename': file_obj.name,
                        'predictedLabel': predicted_label,
                        'confidence': f'{confidence:.2f}%'
                    })
                
                # Clean up - delete the uploaded file
                fs.delete(filename)
            
            except Exception as file_error:
                results.append({
                    'filename': file_obj.name,
                    'error': str(file_error)
                })
        
        # Return JSON response with all predictions
        return JsonResponse({
            'success': len(results) > 0,
            'results': results
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)



# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# from django.http import JsonResponse
# from PIL import Image
# import os
# from django.conf import settings
# from django.views.decorators.csrf import csrf_exempt

# # Mock class labels
# class_labels = {0: "Smartphone", 1: "Laptop"}

# def index(request):
#     return render(request, 'index.html', {'title': 'Image Classifier'})

# def predictImage(request):
#     context = {'title': 'Image Classifier'}
    
#     if request.method == 'POST' and request.FILES.getlist('filePath'):
#         try:
#             results = []
#             files = request.FILES.getlist('filePath')
#             fs = FileSystemStorage()
            
#             for file_obj in files:
#                 # Validate file type
#                 if not file_obj.name.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
#                     context['error'] = 'Please upload only image files (PNG, JPG, JPEG, WEBP)'
#                     continue
                
#                 # Save file
#                 filename = fs.save(file_obj.name, file_obj)
#                 file_url = fs.url(filename)
                
#                 # Mock prediction (alternates between laptop and smartphone)
#                 predicted_class = len(results) % 2  # Alternates between 0 and 1
#                 predicted_label = class_labels[predicted_class]
#                 confidence = 95.5  # Mock confidence value
                
#                 results.append({
#                     'filePathName': file_url,
#                     'predictedLabel': predicted_label,
#                     'confidence': f'{confidence:.2f}%'
#                 })
            
#             context.update({
#                 'results': results,
#                 'success': True
#             })
                
#         except Exception as e:
#             context['error'] = f'Error processing image: {str(e)}'
    
#     return render(request, 'index.html', context)

# @csrf_exempt
# def predict_api(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
#     if not request.FILES:
#         return JsonResponse({'error': 'No files uploaded'}, status=400)
    
#     results = []
#     try:
#         fs = FileSystemStorage()
        
#         # Iterate through all uploaded files
#         for file_key in request.FILES:
#             file_obj = request.FILES[file_key]
            
#             # Validate file type
#             if not file_obj.name.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
#                 results.append({
#                     'filename': file_obj.name,
#                     'error': 'Invalid file type'
#                 })
#                 continue
            
#             try:
#                 # Save file
#                 filename = fs.save(file_obj.name, file_obj)
                
#                 # Mock prediction (alternates between laptop and smartphone)
#                 predicted_class = len(results) % 2  # Alternates between 0 and 1
#                 predicted_label = class_labels[predicted_class]
#                 confidence = 95.5  # Mock confidence value
                
#                 # Add prediction result
#                 results.append({
#                     'filename': file_obj.name,
#                     'predictedLabel': predicted_label,
#                     'confidence': f'{confidence:.2f}%'
#                 })
                
#                 # Clean up - delete the uploaded file
#                 fs.delete(filename)
            
#             except Exception as file_error:
#                 results.append({
#                     'filename': file_obj.name,
#                     'error': str(file_error)
#                 })
        
#         # Return JSON response with all predictions
#         return JsonResponse({
#             'success': len(results) > 0,
#             'results': results
#         })
    
#     except Exception as e:
#         return JsonResponse({
#             'success': False,
#             'error': str(e)
#         }, status=500)
