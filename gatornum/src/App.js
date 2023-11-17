import './App.css';
import React, { useState } from 'react';

const OCRApp = () => {
  const [transcribedText, setTranscribedText] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);

  // Function to handle image selection
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    // You can perform additional checks here, e.g., file type validation
    setSelectedImage(URL.createObjectURL(file));
    // Here, you can integrate your OCR functionality to transcribe text from the image
    // Once you have the transcribed text, update the state using setTranscribedText
    // setTranscribedText(transcribedTextFromOCR);
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-gray-800 text-white p-4">
        <h1 className="text-2xl font-bold">GatorNum</h1>
      </header>

      {/* Main Content */}
      <main
        className="flex-1 p-4 bg-cover bg-opacity bg-gradient-to-t from-gray-700 to-gray-900"
        style={{
          /* Credit: https://www.pexels.com/photo/blue-and-purple-cosmic-sky-956999/  */
          // backgroundImage: `url('/images/bg_image.jpg')`, // Add your image path here
        }}
      >
        <div className="mb-4">
          <h2 className="text-xl text-center font-bold mb-2 text-white">Description</h2>
          <p className="text-white text-center">Your description goes here...</p>
        </div>

        {/* Image Upload */}
        <div className="flex flex-col items-center justify-center">
        <div className="bg-white p-4 rounded-md">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="border border-gray-300 p-2 rounded-md"
          />
        </div>
        </div>

        {/* Image Placeholder */}
        {selectedImage && (
          <div className="mb-4">
            <img
              src={selectedImage}
              alt="Selected"
              className="max-w-full h-auto rounded-md shadow-md"
            />
          </div>
        )}

        {/* Transcribed Text Box */}
        <div>
          <h2 className="text-xl font-bold mb-2 text-center text-white p-2">Transcribed Text</h2>
          <textarea
            value={transcribedText}
            onChange={(e) => setTranscribedText(e.target.value)}
            className="border border-gray-300 p-2 rounded-md w-full h-32 resize-none"
            placeholder="Transcribed text will appear here..."
            readOnly
          ></textarea>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4">
        <p>&copy; 2023 Gabriel Castejon & Larry Mason</p>
      </footer>
    </div>
  );
};

export default OCRApp;