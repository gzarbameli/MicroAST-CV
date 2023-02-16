import { IonContent, IonHeader, IonPage, IonIcon, IonToolbar, IonItem, IonCard, IonLabel, IonButton, IonCardTitle, IonCardHeader, IonCardSubtitle } from '@ionic/react';
import ExploreContainer from '../components/ExploreContainer';
import './Tab2.css';
import React, { useRef, useState, useEffect, useCallback } from "react";
import { Gallery, Item } from 'react-photoswipe-gallery'
import Tab1 from './Tab1';
import IonPhotoViewer from '@codesyntax/ionic-react-photo-viewer';
import ImageViewer from 'react-simple-image-viewer';

const Tab2: React.FC = () => {

  const [selectedImage, setSelectedImage] = useState<string[]>([]);
  const [isViewerOpen, setIsViewerOpen] = useState(false);

  const galleryImages = [
    {
      url: 'http://localhost:5000/gallery/5.jpeg',
      title: 'Kandinskij',
      subtitle: 'Original',
      style: 'http://localhost:5000/predefined_styles/kandinskij.jpg',
      content: 'http://localhost:5000/gallery/1.jpg'
    },
    {
      url: 'http://localhost:5000/gallery/3.jpeg',
      title: 'Picasso',
      subtitle: 'dec-tuned-style6-ssc6',
      style: 'http://localhost:5000/predefined_styles/picasso.jpg',
      content: 'http://localhost:5000/gallery/1.jpg'
    },
    {
      url: 'http://localhost:5000/gallery/2.jpeg',
      title: 'Classic',
      subtitle: 'Original',
      style: 'http://localhost:5000/predefined_styles/classic.jpg',
      content: 'http://localhost:5000/gallery/1.jpg'
    },
    {
      url: 'http://localhost:5000/gallery/4.jpeg',
      title: 'Monet',
      subtitle: 'Original',
      style: 'http://localhost:5000/predefined_styles/monet.jpg',
      content: 'http://localhost:5000/gallery/1.jpg'
    }
  ]

  const openViewer = useCallback((img: any) => {
    setSelectedImage([img]);
    setIsViewerOpen(true);
  }, []);

  const closeViewer = () => {
    setSelectedImage([]);
    setIsViewerOpen(false);
  };

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
        <IonButton className='action1' fill="clear" onClick={() => openViewer(img.style)}>Style</IonButton>
        <IonButton className='action2' fill="clear" onClick={() => openViewer(img.content)}>Content</IonButton>
      </IonCardHeader>
       </IonCard>
      ))}

      {isViewerOpen && (
        <ImageViewer
          src={ selectedImage }
          disableScroll={ false }
          closeOnClickOutside={ true }
          onClose={ closeViewer }
          backgroundStyle={{
            backgroundColor: "rgba(0,0,0,0.5)"
      }}
        />
      )}  

      </IonContent>
      </IonPage>
    );
  }

export default Tab2;