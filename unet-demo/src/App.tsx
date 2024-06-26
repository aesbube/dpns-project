import React, { useState } from 'react';
import axios from 'axios';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';

type TextAlign = 'left' | 'right' | 'center' | 'justify';

const App: React.FC = () => {
  const [result_image, setResult] = useState<string | null>(null);
  const [selected_image, setSelectedImage] = useState<File | null>(null);

  const styles = {
    title: {
      color: '#e8e6e3',
    },
    app: {
      textAlign: 'center' as TextAlign,
      paddingTop: '20px',
      width: '100%',
      height: '100vh',
      backgroundColor: '#1c1f20',
    },
    button: {
      marginTop: '10px',
      padding: '10px 20px',
      borderRadius: '20px',
      backgroundColor: '#303436',
      color: '#e8e6d8',
      border: 'none',
      cursor: 'pointer',
      marginBottom: `20px`,
    },
    buttonHover: {
      backgroundColor: '#45a049',
    }
  };

  const handleImageUpload = async () => {
    if (!selected_image) {
      console.error('No image selected.');
      return;
    }
    const formData = new FormData();
    formData.append('file', selected_image);

    try {
      const response = await axios.post('http://localhost:8000/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(`data:image/png;base64,${response.data.result_image}`);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  const handleSubmit = () => {
    handleImageUpload();
  };

  const handleImageSelect = (image: File) => {
    setSelectedImage(image);
  };

  const handleDownloadImage = () => {
    if (result_image) {
      const download_link = document.createElement('a');
      download_link.href = result_image;
      download_link.download = 'result_image.png'; 
      download_link.click(); 
    } else {
      console.error('No result image available for download.');
    }
  };

  return (
    <div style={styles.app}>
      <h1 style={styles.title}>U-Net Model Demo</h1>
      <ImageUpload onImageUpload={handleImageSelect} />
      <button onClick={handleSubmit} style={styles.button}>Submit</button>
      {result_image && <ResultDisplay resultImage={result_image} />}
      {result_image && <button onClick={handleDownloadImage} style={styles.button}>Download Image</button>}
    </div>
  );
};

export default App;