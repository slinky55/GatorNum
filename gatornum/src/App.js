import './App.css';
import React, { useState } from 'react';
import axios from 'axios';



const OCRApp = () => {
  const [labeledOutput, setLabeledOutput] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  // const [imageData, setImageData] = useState(null);
  const [base64, setBase64] = useState(null);

  // Function to handle image selection
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];


    // You can perform additional checks here, e.g., file type validation
    // console.log(file);
    setSelectedImage(URL.createObjectURL(file));
    setBase64(await convertToBase64(file));
    // console.log(selectedImage);

    // setImageBlob(await image.blob()); // Convert the fetched data into a Blob object
    // console.log(imageBlob);
  };

  const convertToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  // Function to handle form submission
  const handleSubmit = () => {
    // Check if an image is selected
    if (selectedImage) {
      // Create a FormData object to send the image file

      const formData = {
        'image': base64
      };
      console.log(formData);

      // Make a POST request to the Flask backend
      axios.post('/scan', formData) // Replace with your backend URL
        .then(response => {
          // Assuming the response contains the transcribed text
          console.log(response.data);
          setLabeledOutput(response.data.labeled); // Set the transcribed text in the state
        })
        .catch(error => {
          console.error('Error:', error);
        });
    } else {
      console.error('No image selected');
    }
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
            <div className="mb-4 flex justify-center" style={{ paddingTop: '40px' }}>
            <img
              src={selectedImage}
              alt="Selected"
              className="max-w-3/4 h-auto rounded-md shadow-md"
            />
          </div>
        )}

        {/* Output Image */}
        {labeledOutput && (
          <div style={{ maxWidth: '75%', margin: '0 auto', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            <h2 className="text-xl font-bold mb-2 text-center text-white p-2">Output Image</h2>
            <img
              src={`data:image/png;base64,${labeledOutput}`}
              alt="Output"
              className="max-w-full h-auto rounded-md shadow-md"
            />
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mx-auto mt-4"
          style={{ display: 'block' }}
        >
          Submit
        </button>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4">
        <p>&copy; 2023 Gabriel Castejon & Larry Mason</p>
      </footer>
    </div>
  );
};

export default OCRApp;