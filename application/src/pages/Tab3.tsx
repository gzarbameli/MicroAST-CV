import { IonContent, IonHeader, IonPage, IonButton, IonToolbar, IonIcon, IonLabel, IonCard, IonCardTitle } from '@ionic/react';
import { logoGithub } from 'ionicons/icons';
import ExploreContainer from '../components/ExploreContainer';
import './Tab3.css';

const openGithub = () => {
  window.open('https://github.com/gzarbameli/MicroAST-CV')
}
const Tab3: React.FC = () => {
  return (
    <IonPage>   
      <IonHeader>
        <IonToolbar>
          <h1 className="title">About the project</h1>
        </IonToolbar>
      </IonHeader>
      
      <IonContent fullscreen>
      <IonCard class="text-card" color="dark">
      <IonCardTitle class="about">Original Model</IonCardTitle>
        <p className='about'>
        <strong>MicroAST</strong> is a lightweight model that completely abandons the use of cumbersome 
        pre-trained Deep Convolutional Neural Networks (e.g., VGG) at inference. 
        Instead, two micro encoders (content and style encoders) and one micro decoder 
        are utilized for style transfer. The content encoder aims at extracting the main 
        structure of the content image. The style encoder, coupled with a modulator, 
        encodes the style image into learnable dual-modulation signals that modulate 
        both intermediate features and convolutional filters of the decoder, thus injecting
        more sophisticated and flexible style signals to guide the stylizations. 
        In addition, to boost the ability of the style encoder to extract more distinct 
        and representative style signals, it also introduces a new style signal contrastive loss.
        MicroAST is 5-73 times smaller and 6-18 times faster than the state of the art, for the 
        first time enabling super-fast (about 0.5 seconds) arbitrary style transfer at 4K ultra-resolutions.
        </p>
        </IonCard>
        <IonCard class="improvements-card" color="dark">
        <IonCardTitle class="about">Our Works</IonCardTitle>
        <dl>
          <dt><strong> 1. Code Refactoring with new libraries:</strong></dt>
          <dd>- PyTorch Lightning</dd>
          <dd>- Torchmetrics</dd>
          <dd>- Weights & Biases</dd>
          <dt><strong>2. Performance improvements through different experiments:</strong></dt>
          <dd>- Architecture</dd>
          <dd>- Loss functions</dd>
          <dt><strong>3. Fine tuning of the model for two different tasks:</strong></dt>
          <dd>- Generic images</dd>
          <dd>- Images with faces</dd>
          <dt><strong>4. Application development for testing the different models</strong></dt>
        </dl>

        </IonCard>
        <IonCard class="credits-card" color="dark">
        <IonCardTitle class="about">Credits</IonCardTitle>
          <IonLabel>
            <h3 className="text">Antonio D'Orazio - 1967788</h3>
          </IonLabel>
          <IonLabel>
            <h3 className="text">Robert Adrian Minut - 1942806</h3>
          </IonLabel>
          <IonLabel>
            <h3 className="text">Giacomo Zarba Meli - 1807439</h3>
          </IonLabel>
        </IonCard>
        <IonButton color="dark" expand="block" onClick={openGithub} style={{width: "320px", height:"7%"}}>
              <IonIcon icon={logoGithub} />      
        </IonButton>
        <br></br>
      </IonContent>
    </IonPage>
  );
};

export default Tab3;
