/* Only public configuration belongs here. Never add API keys or your email. */
window.FER_CONFIG = {
  formspreeEndpoint: '', // https://formspree.io/f/YOUR_FORM_ID
  maxUploadMB: 20,
  defaultModel: 'cnn',
  models: [
    { id: 'cnn', name: 'Custom CNN', manifest: './models/cnn.json' },
    { id: 'resnet', name: 'ResNet50', manifest: './models/resnet.json' },
    { id: 'mobilenet', name: 'MobileNetV2', manifest: './models/mobilenet.json' },
    { id: 'vgg16', name: 'VGG16', manifest: './models/vgg16.json' }
  ]
};
