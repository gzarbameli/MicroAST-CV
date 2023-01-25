import { IonContent, IonHeader, IonPage, IonIcon, IonToolbar, IonCard, IonButton, useIonViewDidEnter } from '@ionic/react';
import ExploreContainer from '../components/ExploreContainer';
import './Tab2.css';
import React, { useRef, useState, useEffect, useContext } from "react";

import Tab1 from './Tab1';

const Tab2: React.FC = () => {

  const [images, setImages] = useState<string[]>([]);

  const fetchImages = async () => {
      try {
          const response = await fetch('http://127.0.0.1:5000/gallery');
          const data = await response.json();
          setImages(data);
      } catch (e) {
          console.log(e);
      }
  }

  useEffect(() => {
      fetchImages();
  }, []);

  return (
    <IonPage>   
        <IonHeader>
          <IonToolbar>
            <h1 className="title">Gallery</h1>
          </IonToolbar>
        </IonHeader>
        <IonContent fullscreen>
        {images.map((img, index) => (
          <IonCard key={index}>
            <img src={`data:image/jpeg;base64,${img}`} alt="Final image" />
          </IonCard>
        ))}
      </IonContent>
    </IonPage>
  );
};

export default Tab2;


