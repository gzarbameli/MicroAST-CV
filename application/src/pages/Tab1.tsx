import {
    IonContent,
    IonHeader,
    IonPage,
    IonTitle,
    IonToolbar,
    IonItem,
    IonLabel,
    IonButton,
    IonIcon,
    IonList, IonSelect, IonSelectOption, IonSpinner, IonLoading
  } from "@ionic/react";
  import React, { useRef, useState } from "react";
  import "./Tab1.css";
  import placeholder from "../images/placeholder.jpg"

  interface StyleValue {
    file: any;
  }

  interface ContentValue {
    file: any;
  }
  
  const openStyleFileDialog = () => {
    (document as any).getElementById("style-file-upload").click();
 };

 const openContentFileDialog = () => {
    (document as any).getElementById("content-file-upload").click();
 };

  const Tab1: React.FC = () => {

    const [finalImage, setFinalImage] = useState("");
    const [isFinalImage, setIsFinalImage] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isStyleLoaded, setIsStyleLoaded] = useState(false);
    const [isContentLoaded, setIsContentLoaded] = useState(false);

    function DownloadButton() {
        return (
          <a href={`data:image/jpeg;base64,${finalImage}`} download>
            <IonButton color="primary" expand="block" style={{width: "280px", height:"30px", marginLeft: "4%"}}>
            DOWNLOAD
            </IonButton>
          </a>
        );
    }
    
    // Single File Upload
    const styleValue = useRef<StyleValue>({
      file: false,
    });
  
    // Single File Upload
    const contentValue = useRef<ContentValue>({
        file: false,
      });

    // Single File Upload
    const onStyleFileChange = (fileChangeEvent: any) => {
      styleValue.current.file = fileChangeEvent.target.files[0];
      setIsStyleLoaded(true);
    };

    const onContentFileChange = (fileChangeEvent: any) => {
        contentValue.current.file = fileChangeEvent.target.files[0];   
        setIsContentLoaded(true);
      };
  
    const submitForm = async () => {
      if (!styleValue.current.file) {
        console.log("Missing style...")
        return false;
      }

      if (!contentValue.current.file) {
        console.log("Missing content...")
        return false;
      }
      
      setIsLoading(true);

      let formData = new FormData();

      formData.append("styleFile", styleValue.current.file, styleValue.current.file.name);
      formData.append("contentFile", contentValue.current.file, contentValue.current.file.name);
      
      try {
        const serverUrl = "http://127.0.0.1:5000/upload";
  
        const response = await fetch(serverUrl, {
          method: "POST",
          body: formData,
        });
  
        if (!response.ok) {
          throw new Error(response.statusText);
        }
  
        const responseText = await response.text();
        setFinalImage(responseText)
        setIsFinalImage(true)
        setIsLoading(false);
        
      } catch (err) {
        console.log(err);
      }
    };
  
    return (
      <IonPage>
        <IonHeader collapse="condense">
          <IonToolbar>
            <IonTitle size="large">Arbitrary Style Transfer</IonTitle>
          </IonToolbar>
        </IonHeader>
        <IonContent fullscreen>

          <br></br>   
          <IonItem>
            <IonLabel>
              <h1 className="text">Select your style image </h1>
              <p>Upload from your device or select a predefined style</p>
            </IonLabel>
          </IonItem>

          <input
            type="file"
            id="style-file-upload"
            style={{ display: "none" }}
            onChange={(ev) => onStyleFileChange(ev)}/>
          <IonButton color="primary" expand="block" onClick={openStyleFileDialog}>
            UPLOAD
          </IonButton>
          
          {isStyleLoaded &&
            <IonLabel color='success'>
              <small>Style Uploaded!</small>
            </IonLabel>
          }
          
          <IonList>
            <IonItem text-center>
              <IonSelect
                  placeholder="Predefined styles"
                  //onIonChange={(e) => pushLog(`ionChange fired with value: ${e.detail.value}`)}
                  //onIonCancel={() => pushLog('ionCancel fired')}
                  //onIonDismiss={() => pushLog('ionDismiss fired')}
                >
                  <IonSelectOption value="Picasso">Picasso</IonSelectOption>
                  <IonSelectOption value="Monet">Monet</IonSelectOption>
                  <IonSelectOption value="Giotto">Giotto</IonSelectOption>
              </IonSelect>
            </IonItem>
          </IonList>

          <br></br>
          <IonItem>
            <IonLabel>
              <h1 className="text">Select your content image </h1>
            </IonLabel>
          </IonItem>

          <input
            type="file"
            id="content-file-upload"
            style={{ display: "none" }}
            onChange={(ev) => onContentFileChange(ev)}/>
          <IonButton color="primary" expand="block" onClick={openContentFileDialog}>
            UPLOAD
          </IonButton>

          {isContentLoaded &&
            <IonLabel color='success'>
              <small>Content Uploaded!</small>
            </IonLabel>
          }

          <br></br>  
          <br></br> 

          <IonButton color="secondary" expand="block" onClick={submitForm}>TRANSFER!</IonButton>
              <IonLoading
                isOpen={isLoading}
                message={"Stylizing..."}
              />
          <br></br>
          
          {!isFinalImage ? (
                <div style={{
                    position: 'absolute',
                    width: '320px',
                    height: '240px',
                    left: '47%',
                    bottom:'120px',
                    marginLeft: '-150px',
                    border: '1px solid black',
                    padding: '2px'
                }}>
                <img src={placeholder} alt="Your image" 
                style={{width: '100%', height: '100%', margin: 'auto'}}/>
                </div>
            ) : (
                <div style={{
                    position: 'absolute',
                    width: '320px',
                    height: '240px',
                    bottom:'120px',
                    left: '47%',
                    marginLeft: '-150px',
                    border: '1px solid black',
                    padding: '2px'
                }}>
                <img src={`data:image/jpeg;base64,${finalImage}`} alt="Your image" 
                style={{width: '100%', height: '100%', margin: 'auto'}}/>
                <DownloadButton />
                </div>
            )}
            
        </IonContent>
      </IonPage>
      
    );
  };
  
  export default Tab1;