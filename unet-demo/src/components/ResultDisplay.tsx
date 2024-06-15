import React from "react";

interface ResultDisplayProps {
    resultImage: string | null;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ resultImage }) => {
    const styles = {
        image: {
            margin: '20px',
            maxWidth: '100%',
            height: 'auto',
            border: '2px solid black',
        }
    };
    
    return (
        <div>
            {resultImage && <img src={resultImage} alt="Result Image" style={styles.image}></img>}
        </div>
    );
};


export default ResultDisplay
