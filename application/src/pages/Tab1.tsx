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
    IonList, IonSelect, IonSelectOption, IonAlert, IonLoading
  } from "@ionic/react";
  import { download, imageOutline, brushOutline } from 'ionicons/icons';
  import React, { useRef, useState, useContext } from "react";
  import "./Tab1.css";
  import placeholder from "../images/placeholder.jpg"
  import Tab2 from "./Tab2";

  interface StyleValue {
    file: any;
  }

  interface ContentValue {
    file: any;
  }
  
  const openStyleFileDialog = () => {
    (document as any).getElementById("style-file-upload").value = null;
    (document as any).getElementById("style-file-upload").click();
    let select = (document as any).getElementById("my-select")
    select.value = "";
 };

 const openContentFileDialog = () => {
    (document as any).getElementById("content-file-upload").value = null;
    (document as any).getElementById("content-file-upload").click();
 };


  const Tab1: React.FC = () => {
    
    const [finalImage, setFinalImage] = useState("");
    const [isFinalImage, setIsFinalImage] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [contentName, setContentName] = useState("");
    const [styleName, setStyleName] = useState("");
    const [predefinedStyle, setPredefinedStyle] = useState("");
    const [showAlert, setShowAlert] = useState(false);
    const [model, setModel] = useState("original");

    function DownloadButton() {
        return (
          <a href={`data:image/jpeg;base64,${finalImage}`} download>
            <IonButton color="success" expand="block" style={{width: "80px", height:"40px", marginTop: '7%', margin: 'auto', display: 'block'}}>
              <IonIcon icon={download} />
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
    const onStyleFileChange = async (fileChangeEvent: any) => {
      styleValue.current.file = await fileChangeEvent.target.files[0];
      setStyleName(styleValue.current.file.name)
      setPredefinedStyle("")
    };

    const onContentFileChange = (fileChangeEvent: any) => {
        contentValue.current.file = fileChangeEvent.target.files[0];   
        setContentName(contentValue.current.file.name)
      };

    const onPredefinedStyleChange = async (value: any) => {
      setPredefinedStyle(await value)
      styleValue.current.file = false
      setStyleName(styleValue.current.file.name)
    };

    const onModelChange = async (value: any) => {
      setModel(await value)
    };
  
    const submitForm = async () => {
      if (!styleValue.current.file && predefinedStyle=="") {
        console.log("Missing style...")
        setShowAlert(true)
        return false;
      }

      if (!contentValue.current.file) {
        console.log("Missing content...")
        setShowAlert(true)
        return false;
      }
      
      setIsLoading(true);

      let formData = new FormData();

      if (predefinedStyle=="") {
        formData.append("styleFile", styleValue.current.file, styleValue.current.file.name);
      }
      
      formData.append("contentFile", contentValue.current.file, contentValue.current.file.name);
      
      try {
        const serverUrl = "http://127.0.0.1:5000/upload";
  
        const response = await fetch(serverUrl, {
          method: "POST",
          body: formData,
          headers: new Headers({"predefinedStyle":predefinedStyle, "model":model})
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
        <IonHeader>
          <IonToolbar>
            <h1 className="title">Arbitrary Style Transfer</h1>
          </IonToolbar>
        </IonHeader>
        <IonContent fullscreen >
          <br></br>   

          <IonItem className="ion-text-center" lines="full">
            <IonLabel>
              <h1 className="text">Select your style image </h1>
              <p>Upload from your device or select a predefined style</p>
            </IonLabel>
          </IonItem>

          <input
            type="file"
            id="style-file-upload"
            style={{ display: "none" }}
            onInput={(ev) => onStyleFileChange(ev)}/>
          <IonButton color="primary" expand="block" onClick={openStyleFileDialog}>
            {!(styleName) ? (
                <IonIcon icon={imageOutline} slot={"icon-only"} />
              ) : (
                styleValue.current.file.name
            )}
          </IonButton>
                
          <IonList>
            <IonItem lines="full">
              <IonSelect
                  id="my-select"
                  placeholder="Predefined styles"
                  value={predefinedStyle}
                  onIonChange={(e) => onPredefinedStyleChange(e.detail.value)}
                >
                  <IonSelectOption value="Picasso">Picasso</IonSelectOption>
                  <IonSelectOption value="Monet">Monet</IonSelectOption>
                  <IonSelectOption value="Kandinskij">Kandinskij</IonSelectOption>
              </IonSelect>
            </IonItem>
          </IonList>

          <br></br>
          <IonItem lines="full">
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
            {!(contentName) ? (
                  <IonIcon icon={imageOutline} slot={"icon-only"} />
                ) : (
                  contentName
              )}
          </IonButton>

          <IonList>
            <IonItem lines="full">
              <IonSelect
                  id="my-select-model"
                  placeholder="Model"
                  onIonChange={(e) => onModelChange(e.detail.value)}
                >
                  <IonSelectOption value="original">original</IonSelectOption>
                  <IonSelectOption value="dec-tuned-style6-ssc6">dec-tuned-style6-ssc6</IonSelectOption>
                  <IonSelectOption value="dec-tuned-coco-wikiart">dec-tuned-coco-wikiart</IonSelectOption>
                  <IonSelectOption value="dec-tuned-finetuning-wikiart-faces-l2">dec-tuned-finetuning-wikiart-faces-l2</IonSelectOption>
                  <IonSelectOption value="dec-tuned-tv-coco-wikiart">dec-tuned-tv-coco-wikiart</IonSelectOption>
                  <IonSelectOption value="dec-tuned-tv-ffhq-laion-scratch">dec-tuned-tv-ffhq-laion-scratch</IonSelectOption>
                  <IonSelectOption value="dec-tuned-tv-finetuning-ffhq-laion-l2">dec-tuned-tv-finetuning-ffhq-laion-l2</IonSelectOption>
                  <IonSelectOption value="dec-tuned-tv-ffhq-wikiart-faces-scratch">dec-tuned-tv-ffhq-wikiart-faces-scratch</IonSelectOption>
                  <IonSelectOption value="dec-tuned-tvfinetuning-ffhq-wikiart-l1">dec-tuned-tvfinetuning-ffhq-wikiart-l1</IonSelectOption>
                  
              </IonSelect>
            </IonItem>
          </IonList>

          <IonButton className="transfer-button" color="secondary" expand="block" onClick={submitForm}>
            <IonIcon icon={brushOutline} slot={"icon-only"}/>
          </IonButton>

          <IonAlert
            isOpen={showAlert}
            onDidDismiss={() => setShowAlert(false)}
            header="Wait..."
            subHeader="Missing Style or Content!"
            buttons={['OK']}
          />
            
          <IonLoading
            isOpen={isLoading}
            message={"Stylizing..."}
          />        
          <br></br> 
          <br></br> 
          <br></br> 
          {!isFinalImage ? (
                <div style={{
                    width: '320px',
                    height: '230px',
                    marginLeft: 'auto',
                    marginRight: 'auto',
                    position: "relative",
                    border: '1px solid black',
                    padding: '2px',
                }}>
                <img src={placeholder} alt="Your image" 
                style={{width: '100%', height: '100%', margin: 'auto'}}/>
                
                </div>
                
            ) : (
                <div style={{
                  width: '320px',
                    height: '260px',
                    marginLeft: 'auto',
                    marginRight: 'auto',
                    position: "relative",
                    padding: '2px'
                }}>
                <img src={`data:image/jpeg;base64,${finalImage}`} alt="Your image" 
                style={{width: '100%', height: '80%', margin: 'auto', border: '1px solid black'}}/>
                <DownloadButton />
                </div>
            )}

        </IonContent>
      </IonPage>
      
    );
  };
  
  export default Tab1;