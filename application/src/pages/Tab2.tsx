import { IonContent, IonHeader, IonPage, IonIcon, IonToolbar, IonItem, IonCard, IonLabel, IonCardContent, IonCardTitle, IonCardHeader, IonCardSubtitle } from '@ionic/react';
import ExploreContainer from '../components/ExploreContainer';
import './Tab2.css';
import React, { useRef, useState, useEffect, useCallback } from "react";
import { Gallery, Item } from 'react-photoswipe-gallery'
import Tab1 from './Tab1';
import IonPhotoViewer from '@codesyntax/ionic-react-photo-viewer';

const Tab2: React.FC = () => {

  //const [images, setImages] = useState<string[]>([]);
//
  //const fetchImages = async () => {
  //    try {
  //        const response = await fetch('http://127.0.0.1:5000/gallery');
  //        const data = await response.json();
  //        const modifiedImages = await data.map((img: string) => `data:image/jpeg;base64,${img}`);
  //        setImages(modifiedImages);
  //    } catch (e) {
  //        console.log(e);
  //    }
  //}

  const galleryImages = [
    {
      url: 'http://localhost:5000/gallery/2.jpeg',
      title: 'Kandinskij',
      subtitle: 'Predefined Style'
    },
    {
      url: 'http://localhost:5000/gallery/3.jpeg',
      title: 'Picasso',
      subtitle: 'Predefined Style'
    },
    {
      url: 'http://localhost:5000/gallery/4.jpeg',
      title: 'Classic',
      subtitle: 'Custom Style'
    },
    {
      url: 'http://localhost:5000/gallery/5.jpeg',
      title: 'Monet',
      subtitle: 'Predefined Style'
    }
  ]

  //useEffect(() => {
  //    fetchImages();
  //}, []);
  //

    return (
      <IonPage>   
      <IonHeader>
        <IonToolbar>
          <h1 className="title">Gallery</h1>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
      {galleryImages.map((img, index) => (
      <IonCard key={index} color="dark">
            <IonPhotoViewer
              title={img.title}
              src={img.url}
              showHeader={false}
            >
                <img
                src={img.url}
                alt={img.title}
              />
            </IonPhotoViewer>
       <IonCardHeader>
        <IonCardSubtitle>{img.subtitle}</IonCardSubtitle>
        <IonCardTitle>{img.title}</IonCardTitle>
        
      </IonCardHeader>
       </IonCard>
      ))}
      </IonContent>
      </IonPage>
    );
  }

export default Tab2;