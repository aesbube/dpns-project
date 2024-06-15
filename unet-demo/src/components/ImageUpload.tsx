import React, { useState } from "react";

interface ImageUploadProps {
    onImageUpload: (image: File) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUpload }) => {
    const [image, setImage] = useState<File | null>(null);

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setImage(event.target.files[0]);
            onImageUpload(event.target.files[0]);
        }
    };
    return (
        <div>
            <input type="file" accept="image/*" onChange={handleImageUpload} style={styles.input} />
        </div>
    );


};

const styles = {
    input: {
        margin: '20px 0',
    }
};

export default ImageUpload;
