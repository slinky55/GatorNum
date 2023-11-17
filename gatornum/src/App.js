import logo from './logo.svg';
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
      <main className="flex-1 p-4">
        <div className="mb-4">
          <h2 className="text-xl font-bold mb-2">Description</h2>
          <p className="text-gray-600">Your description goes here...</p>
        </div>

        {/* Image Upload */}
        <div className="mb-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="border border-gray-300 p-2 rounded-md"
          />
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
          <h2 className="text-xl font-bold mb-2">Transcribed Text</h2>
          <textarea
            value={transcribedText}
            onChange={(e) => setTranscribedText(e.target.value)}
            className="border border-gray-300 p-2 rounded-md w-full h-32 resize-none"
            placeholder="Transcribed text will appear here..."
          ></textarea>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4">
        <p>&copy; 2023 Your Company</p>
      </footer>
    </div>
  );
};

export default OCRApp;