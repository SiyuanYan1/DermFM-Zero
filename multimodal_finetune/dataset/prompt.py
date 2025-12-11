from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from typing import Dict, List

class Derm7ptPromptBuilder:
    """
    A class to build concise prompts for VLM dermatology classification
    using "This is a patient with {xxx}" format.
    """
    
    def __init__(self, meta_df: pd.DataFrame):
        """
        Initialize the prompt builder with metadata DataFrame.
        
        Args:
            meta_df: DataFrame containing metadata with columns:
                    ['elevation', 'sex', 'location']
        """
        self.meta_df = meta_df
        self.feature_mappings = self._create_feature_mappings()
    
    def _create_feature_mappings(self) -> Dict:
        """Create mappings for each metadata feature."""
        mappings = {}
        
        for feat in ['elevation', 'sex', 'location']:
            if feat in self.meta_df.columns:
                unique_vals = self.meta_df[feat].unique()
                unique_vals = [val for val in unique_vals if pd.notna(val)]
                mappings[feat] = sorted(unique_vals)
        
        return mappings
    
    def build_patient_description(self, index: int) -> str:
        """Build patient description from metadata."""
        if index >= len(self.meta_df):
            return "This is a patient with a skin lesion."
        
        sample_meta = self.meta_df.iloc[index]
        
        # Build description components
        components = []
        
        # Add sex
        if 'sex' in sample_meta and pd.notna(sample_meta['sex']):
            components.append(f"{sample_meta['sex']}")
        
        # Add elevation type
        if 'elevation' in sample_meta and pd.notna(sample_meta['elevation']):
            components.append(f"{sample_meta['elevation']} lesion")
        
        # Add location
        if 'location' in sample_meta and pd.notna(sample_meta['location']):
            location = sample_meta['location']
            if location == 'head neck':
                location = 'head and neck'
            elif location == 'upper limbs':
                location = 'upper limbs'
            elif location == 'lower limbs':
                location = 'lower limbs'
            elif location == 'genital areas':
                location = 'genital area'
            components.append(f"on {location}")
        
        if components:
            description = f"This is a {' '.join(components)}."
        else:
            description = "This is a patient with a skin lesion."
        
        return description

class MILKPromptBuilder:
    """
    A class to build concise prompts for VLM dermatology classification
    using "This is a patient with {xxx}" format for age/sex/site dataset.
    """
    
    def __init__(self, meta_df: pd.DataFrame):
        """
        Initialize the prompt builder with metadata DataFrame.
        
        Args:
            meta_df: DataFrame containing metadata with columns:
                    ['age', 'sex', 'site']
        """
        self.meta_df = meta_df
        self.feature_mappings = self._create_feature_mappings()
    
    def _create_feature_mappings(self) -> Dict:
        """Create mappings for each metadata feature."""
        mappings = {}
        
        for feat in ['age', 'sex', 'site']:
            if feat in self.meta_df.columns:
                unique_vals = self.meta_df[feat].unique()
                unique_vals = [val for val in unique_vals if pd.notna(val)]
                
                if feat == 'age':
                    # Sort ages numerically
                    mappings[feat] = sorted([float(val) for val in unique_vals])
                else:
                    mappings[feat] = sorted(unique_vals)
        
        return mappings
    
    def _format_age(self, age) -> str:
        """Format age for description."""
        if pd.isna(age):
            return None
        
        age = float(age)
        if age == 0:
            return "newborn"
        elif age <= 2:
            return f"{int(age)}-year-old infant"
        elif age <= 12:
            return f"{int(age)}-year-old child"
        elif age <= 17:
            return f"{int(age)}-year-old adolescent"
        elif age <= 64:
            return f"{int(age)}-year-old adult"
        else:
            return f"{int(age)}-year-old elderly"
    
    def _format_sex(self, sex) -> str:
        """Format sex for description."""
        if pd.isna(sex):
            return None
        
        if sex.lower() == 'm':
            return "male"
        elif sex.lower() == 'f':
            return "female"
        else:
            return sex
    
    def _format_site(self, site) -> str:
        """Format anatomical site for description."""
        if pd.isna(site):
            return None
        
        site_mappings = {
            'head_neck_face': 'head, neck, and face',
            'trunk': 'trunk',
            'upper_extremity': 'upper extremity',
            'lower_extremity': 'lower extremity',
            'hand': 'hand',
            'foot': 'foot',
            'genital': 'genital area'
        }
        
        return site_mappings.get(site, site)
    
    def build_patient_description(self, index: int, MONET_feat=False) -> str:
        """Build patient description from metadata."""
        if index >= len(self.meta_df):
            return "This is a patient with a skin lesion."
        
        sample_meta = self.meta_df.iloc[index]
        
        # Build description components
        components = []
        
        # Add age and sex together for natural flow
        age_desc = self._format_age(sample_meta.get('age'))
        sex_desc = self._format_sex(sample_meta.get('sex'))
        
        if age_desc and sex_desc:
            # Combine age and sex naturally
            if "male" in sex_desc or "female" in sex_desc:
                components.append(age_desc.replace("adult", sex_desc).replace("elderly", f"{sex_desc} elderly patient"))
            else:
                components.append(f"{age_desc} {sex_desc}")
        elif age_desc:
            components.append(age_desc)
        elif sex_desc:
            components.append(f"{sex_desc} patient")
        
        # Add site
        site_desc = self._format_site(sample_meta.get('site'))
        if site_desc:
            components.append(f"with a lesion on the {site_desc}")
        else:
            components.append("with a skin lesion")
        
        if components:
            # Clean up the description
            description = "This is a " + " ".join(components) + "."
            # Fix any double spaces or awkward phrasing
            description = description.replace("  ", " ")
            description = description.replace("a a ", "a ")
        else:
            description = "This is a patient with a skin lesion."

        if MONET_feat:
            exists_concept = []
            concept_mapping = {
                'ulceration_crust': 'ulceration crust',
                'vasculature_vessels': 'vasculature vessels', 
                'erythema': 'erythema',
                'pigmented': 'pigmentation'
            }
        
            for concept in ['ulceration_crust', 'vasculature_vessels', 'erythema', 'pigmented']:
                if sample_meta.get(concept):  
                    exists_concept.append(concept_mapping[concept])
        
            if exists_concept:
                if len(exists_concept) == 1:
                    monet_desc = f" {exists_concept[0]} is likely present on the lesion."
                elif len(exists_concept) == 2:
                    monet_desc = f" {exists_concept[0]} and {exists_concept[1]} are likely present on the lesion."
                else:
                    monet_desc = f" {', '.join(exists_concept[:-1])}, and {exists_concept[-1]} are likely present on the lesion."

                description += monet_desc
        
        return description

class PADPromptBuilder:
    """
    A class to build concise prompts for VLM dermatology classification
    using "This is a patient with {xxx}" format for PAD dataset.
    """
    
    def __init__(self, meta_df: pd.DataFrame):
        """
        Initialize the prompt builder with metadata DataFrame.
        
        Args:
            meta_df: DataFrame containing PAD dataset metadata
        """
        self.meta_df = meta_df
        self.feature_mappings = self._create_feature_mappings()
    
    def _create_feature_mappings(self) -> Dict:
        """Create mappings for each metadata feature."""
        mappings = {}
        
        features = ['smoke', 'drink','age', 'pesticide', 'gender', 'skin_cancer_history',
                   'cancer_history', 'fitspatrick', 'region', 
                   'diameter_1', 'diameter_2', 'itch', 'grew', 'hurt', 
                   'changed', 'bleed', 'elevation']
        
        for feat in features:
            if feat in self.meta_df.columns:
                unique_vals = self.meta_df[feat].unique()
                unique_vals = [val for val in unique_vals if pd.notna(val)]
                
                if feat == 'age':
                    mappings[feat] = sorted([int(val) for val in unique_vals])
                elif feat in ['diameter_1', 'diameter_2', 'fitspatrick']:
                    mappings[feat] = sorted([float(val) for val in unique_vals])
                else:
                    mappings[feat] = sorted(unique_vals)
        
        return mappings
    
    def _format_age_gender(self, age, gender) -> str:
        """Format age and gender together."""
        age_str = ""
        gender_str = ""
        
        if pd.notna(age):
            age = int(age)
            if age <= 12:
                age_str = f"{age}-year-old child"
            elif age <= 17:
                age_str = f"{age}-year-old adolescent"
            elif age <= 64:
                age_str = f"{age}-year-old adult"
            else:
                age_str = f"{age}-year-old elderly patient"
        
        if pd.notna(gender):
            if gender.upper() == 'MALE':
                gender_str = "male"
            elif gender.upper() == 'FEMALE':
                gender_str = "female"
        
        if age_str and gender_str:
            if "patient" in age_str:
                return age_str.replace("patient", f"{gender_str} patient")
            else:
                return f"{age_str.replace('adult', gender_str).replace('elderly', f'{gender_str} elderly')}"
        elif age_str:
            return age_str
        elif gender_str:
            return f"{gender_str} patient"
        else:
            return "patient"
    
    def _format_region(self, region) -> str:
        """Format anatomical region."""
        if pd.isna(region):
            return None
        
        region_map = {
            'FOREARM': 'forearm',
            'CHEST': 'chest',
            'ARM': 'arm',
            'FACE': 'face',
            'BACK': 'back',
            'NOSE': 'nose',
            'THIGH': 'thigh',
            'HAND': 'hand',
            'ABDOMEN': 'abdomen',
            'EAR': 'ear',
            'NECK': 'neck',
            'LIP': 'lip',
            'SCALP': 'scalp',
            'FOOT': 'foot'
        }
        return region_map.get(region, region.lower())
    
    def _format_lesion_characteristics(self, row) -> List[str]:
        """Format lesion characteristics."""
        characteristics = []
        
        # Size information
        if pd.notna(row.get('diameter_1')) or pd.notna(row.get('diameter_2')):
            d1 = row.get('diameter_1', 0) if pd.notna(row.get('diameter_1')) else 0
            d2 = row.get('diameter_2', 0) if pd.notna(row.get('diameter_2')) else 0
            max_diameter = max(d1, d2) if d1 > 0 or d2 > 0 else 0
            if max_diameter > 0:
                characteristics.append(f"{max_diameter}mm")
        
        # Elevation
        if pd.notna(row.get('elevation')) and str(row.get('elevation')).upper() == 'TRUE':
            characteristics.append("elevated")
        
        # Symptomatic features
        symptoms = []
        if pd.notna(row.get('itch')) and str(row.get('itch')).upper() == 'TRUE':
            symptoms.append("itchy")
        if pd.notna(row.get('hurt')) and str(row.get('hurt')).upper() == 'TRUE':
            symptoms.append("painful")
        if pd.notna(row.get('bleed')) and str(row.get('bleed')).upper() == 'TRUE':
            symptoms.append("bleeding")
        
        if symptoms:
            characteristics.append(" and ".join(symptoms))
        
        # Growth/change
        changes = []
        if pd.notna(row.get('grew')) and str(row.get('grew')).upper() == 'TRUE':
            changes.append("growing")
        if pd.notna(row.get('changed')) and str(row.get('changed')).upper() == 'TRUE':
            changes.append("changing")
        
        if changes:
            characteristics.append(" and ".join(changes))
        
        return characteristics
    
    def _format_risk_factors(self, row) -> List[str]:
        """Format risk factors."""
        risk_factors = []
        
        # Smoking and drinking
        lifestyle = []
        if pd.notna(row.get('smoke')) and row.get('smoke'):
            lifestyle.append("smoking history")
        if pd.notna(row.get('drink')) and row.get('drink'):
            lifestyle.append("alcohol consumption")
        if lifestyle:
            risk_factors.append(" and ".join(lifestyle))
        
        # Family history
        family_history = []
        if pd.notna(row.get('skin_cancer_history')) and row.get('skin_cancer_history'):
            family_history.append("family history of skin cancer")
        if pd.notna(row.get('cancer_history')) and row.get('cancer_history'):
            family_history.append("family history of cancer")
        if family_history:
            risk_factors.append(" and ".join(family_history))
        
        # Environmental exposure
        if pd.notna(row.get('pesticide')) and row.get('pesticide'):
            risk_factors.append("pesticide exposure")
        
        # Fitzpatrick skin type
        if pd.notna(row.get('fitspatrick')):
            fitz = int(row.get('fitspatrick'))
            risk_factors.append(f"Fitzpatrick skin type {fitz}")
        
        return risk_factors
    
    def build_patient_description(self, index: int) -> str:
        """Build comprehensive patient description from metadata."""
        if index >= len(self.meta_df):
            return "This is a patient with a skin lesion."
        
        row = self.meta_df.iloc[index]
        
        # Start with basic demographics
        patient_desc = self._format_age_gender(row.get('age'), row.get('gender'))
        
        # Add lesion location
        region = self._format_region(row.get('region'))
        if region:
            lesion_desc = f"lesion on the {region}"
        else:
            lesion_desc = "skin lesion"
        
        # Add lesion characteristics
        characteristics = self._format_lesion_characteristics(row)
        if characteristics:
            char_str = ", ".join(characteristics)
            lesion_desc = f"{char_str} {lesion_desc}"
        
        # Build main description
        main_desc = f"This is a {patient_desc} with a {lesion_desc}"
        
        # Add risk factors if space allows (keep concise)
        risk_factors = self._format_risk_factors(row)
        if risk_factors and len(risk_factors) <= 2:
            risk_str = " and ".join(risk_factors[:2])
            main_desc += f" and {risk_str}"
        
        return main_desc + "."