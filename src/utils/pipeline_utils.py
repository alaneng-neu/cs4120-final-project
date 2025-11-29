import pandas as pd


# User-friendly descriptions for each phishing type
PHISHING_DESCRIPTIONS = {
    'legitimate': {
        'label': '✅ Legitimate Email',
        'description': 'This appears to be a legitimate email.',
        'risk_level': 'Low',
        'advice': 'This email appears safe, but always verify sender addresses.'
    },
    'credential_harvesting': {
        'label': '🔐 Credential Harvesting',
        'description': 'This email attempts to steal your login credentials (username, password).',
        'risk_level': 'High',
        'advice': 'NEVER click links asking for passwords. Go directly to the official website instead.'
    },
    'urgency': {
        'label': '⏰ Urgency Scam',
        'description': 'This email creates false urgency to pressure you into quick action.',
        'risk_level': 'High',
        'advice': 'Legitimate organizations rarely demand immediate action. Take your time and verify.'
    },
    'financial_scam': {
        'label': '💰 Financial Scam',
        'description': 'This email attempts to steal your money or financial information.',
        'risk_level': 'High',
        'advice': 'Never send money or share bank details based on an email request.'
    },
    'authority_scam': {
        'label': '👔 Authority Scam',
        'description': 'This email impersonates a government agency, bank, or authority figure.',
        'risk_level': 'High',
        'advice': 'Government agencies rarely contact you by email. Call official numbers to verify.'
    },
    'romance_dating': {
        'label': '💕 Romance Scam',
        'description': 'This email is from someone trying to build a fake romantic relationship to exploit you.',
        'risk_level': 'Medium',
        'advice': 'Be wary of online relationships, especially if they ask for money or personal info.'
    },
    'generic_phishing': {
        'label': '🎣 Generic Phishing',
        'description': 'This email contains general phishing tactics to deceive you.',
        'risk_level': 'Medium',
        'advice': 'Do not click links or download attachments from unknown senders.'
    },
    'threats': {
        'label': '⚠️ Threat/Extortion',
        'description': 'This email contains threats or extortion attempts.',
        'risk_level': 'High',
        'advice': 'Do not respond to threats. Report to authorities if you feel unsafe.'
    },
    'tech_support': {
        'label': '🖥️ Tech Support Scam',
        'description': 'This email falsely claims your computer has problems and offers fake support.',
        'risk_level': 'High',
        'advice': 'Real tech companies never send unsolicited support emails. Do not call numbers in emails.'
    },
    'social_engineering_advanced': {
        'label': '🧠 Advanced Social Engineering',
        'description': 'This is a sophisticated attack using psychological manipulation.',
        'risk_level': 'High',
        'advice': 'Be skeptical of emails that seem too personal or reference specific details about you.'
    },
    'social_engineering': {
        'label': '🎭 Social Engineering',
        'description': 'This email uses social manipulation tactics to deceive you.',
        'risk_level': 'Medium',
        'advice': 'Verify requests through official channels before taking action.'
    }
}

# Test examples covering different scenarios
test_emails = [
    # Legitimate email
    """Hi John,
    
    Just wanted to follow up on our meeting yesterday. I've attached the quarterly report 
    as discussed. Let me know if you have any questions.
    
    Best regards,
    Sarah""",
    
    # Credential harvesting
    """URGENT: Your account has been compromised!
    
    We detected suspicious activity on your account. Click the link below to verify 
    your password and secure your account immediately:
    
    [Verify Your Account Now]
    
    If you don't verify within 24 hours, your account will be suspended.""",
    
    # Financial scam
    """Congratulations! You have been selected as the winner of our international lottery!
    
    You have won $5,000,000 USD! To claim your prize, please send us your bank account 
    details and a processing fee of $500.
    
    Contact our claims agent immediately!""",
    
    # Authority scam (IRS)
    """FINAL NOTICE: IRS Tax Violation
    
    This is the Internal Revenue Service. Our records show you owe $15,000 in back taxes.
    Failure to pay immediately will result in arrest and seizure of assets.
    
    Call this number now to avoid legal action: 1-800-FAKE-IRS""",
    
    # Tech support scam
    """ALERT: Your computer has been infected!
    
    We have detected a serious virus on your computer. Your personal data is at risk.
    Call Microsoft Support immediately at 1-800-FAKE-HELP to remove the virus.
    
    Do not turn off your computer!"""
]


def classify_email(email_text, spam_model, spam_vectorizer, phishing_model, phishing_vectorizer):
    """
    Two-stage email classification pipeline.
    
    Stage 1: Classify as Ham or Spam
    Stage 2: If Spam, classify the phishing type
    
    Handles conflicts when Stage 1 says Spam but Stage 2 says Legitimate:
    - Compares confidence scores
    - Uses 2nd best prediction from Stage 2 as fallback
    - Shows uncertainty warnings to user
    
    Returns:
        dict: Classification results with labels, confidence, and advice
    """
    result = {
        'email_text': email_text[:200] + '...' if len(email_text) > 200 else email_text,
        'stage1_result': None,
        'stage1_confidence': None,
        'stage2_result': None,
        'stage2_confidence': None,
        'stage2_alternative': None,
        'stage2_alternative_confidence': None,
        'final_classification': None,
        'risk_level': None,
        'advice': None,
        'has_conflict': False,
        'uncertainty_warning': None
    }
    
    # Stage 1: Spam Detection
    email_vec_spam = spam_vectorizer.transform([email_text])
    spam_pred = spam_model.predict(email_vec_spam)[0]
    spam_proba = spam_model.predict_proba(email_vec_spam)[0]
    spam_confidence = spam_proba[spam_pred]
    
    result['stage1_result'] = 'Spam' if spam_pred == 1 else 'Ham'
    result['stage1_confidence'] = spam_confidence
    
    # If Ham (legitimate), stop here
    if spam_pred == 0:
        result['final_classification'] = '✅ Legitimate Email (Ham)'
        result['risk_level'] = 'Low'
        result['advice'] = 'This email appears to be legitimate, but always verify sender addresses.'
        return result
    
    # Stage 2: Phishing Type Classification (only for spam emails)
    email_vec_phishing = phishing_vectorizer.transform([email_text])
    phishing_pred = phishing_model.predict(email_vec_phishing)[0]
    phishing_proba = phishing_model.predict_proba(email_vec_phishing)[0]
    
    # Get all predictions sorted by confidence
    classes = list(phishing_model.classes_)
    sorted_indices = phishing_proba.argsort()[::-1]  # Descending order
    
    top_pred = classes[sorted_indices[0]]
    top_confidence = phishing_proba[sorted_indices[0]]
    
    # Get 2nd best prediction
    second_pred = classes[sorted_indices[1]] if len(sorted_indices) > 1 else None
    second_confidence = phishing_proba[sorted_indices[1]] if len(sorted_indices) > 1 else None
    
    result['stage2_result'] = top_pred
    result['stage2_confidence'] = top_confidence
    result['stage2_alternative'] = second_pred
    result['stage2_alternative_confidence'] = second_confidence
    
    # Check for conflict: Stage 1 says Spam, but Stage 2 says Legitimate
    is_conflict = (top_pred == 'legitimate')
    result['has_conflict'] = is_conflict
    
    if is_conflict:
        # CONFLICT RESOLUTION LOGIC
        # Compare Stage 1 spam confidence vs Stage 2 legitimate confidence
        
        # Calculate confidence difference
        confidence_diff = abs(spam_confidence - top_confidence)
        
        # Determine which prediction to trust more
        if spam_confidence > top_confidence:
            # Trust Stage 1 (Spam) - use 2nd best phishing type
            if second_pred and second_pred != 'legitimate':
                # Use 2nd best prediction as fallback
                result['uncertainty_warning'] = (
                    f"⚠️ CONFLICTING PREDICTIONS: Our spam detector flagged this as spam "
                    f"({spam_confidence:.1%} confident), but the phishing classifier suggested "
                    f"'legitimate' ({top_confidence:.1%} confident). "
                    f"Showing next most likely phishing type instead."
                )
                
                if second_pred in PHISHING_DESCRIPTIONS:
                    info = PHISHING_DESCRIPTIONS[second_pred]
                    result['final_classification'] = f"⚠️ {info['label']} (Uncertain)"
                    result['risk_level'] = 'Medium-High'
                    result['advice'] = (
                        f"{info['advice']} "
                        f"Note: There is some uncertainty in this classification. "
                        f"Exercise extra caution."
                    )
                else:
                    result['final_classification'] = f'⚠️ Suspicious Email ({second_pred})'
                    result['risk_level'] = 'Medium-High'
                    result['advice'] = 'This email has conflicting signals. Treat with caution.'
            else:
                # No good alternative, show generic spam warning
                result['uncertainty_warning'] = (
                    f"⚠️ CONFLICTING PREDICTIONS: Spam detector says spam ({spam_confidence:.1%}), "
                    f"but phishing classifier says legitimate ({top_confidence:.1%}). "
                    f"Treating as suspicious due to spam detection."
                )
                result['final_classification'] = '⚠️ Suspicious Email (Unclassified Spam)'
                result['risk_level'] = 'Medium'
                result['advice'] = (
                    'This email was flagged as spam but could not be categorized. '
                    'Exercise caution and verify the sender.'
                )
        else:
            # Stage 2 (Legitimate) has higher confidence than Stage 1 (Spam)
            # Show as potentially legitimate but with warning
            if confidence_diff > 0.2:
                # Large confidence gap - lean towards legitimate with mild warning
                result['uncertainty_warning'] = (
                    f"⚠️ MIXED SIGNALS: Initially flagged as spam ({spam_confidence:.1%}), "
                    f"but phishing analysis suggests legitimate ({top_confidence:.1%}). "
                    f"Likely safe, but verify sender."
                )
                result['final_classification'] = '⚠️ Likely Legitimate (Verify Sender)'
                result['risk_level'] = 'Low-Medium'
                result['advice'] = (
                    'This email shows mixed signals. It may be legitimate, but was initially '
                    'flagged as suspicious. Double-check the sender address and any links before clicking.'
                )
            else:
                # Close confidence scores - show uncertainty
                result['uncertainty_warning'] = (
                    f"⚠️ UNCERTAIN CLASSIFICATION: Spam detector ({spam_confidence:.1%}) and "
                    f"phishing classifier ({top_confidence:.1%}) disagree. "
                    f"Proceed with caution."
                )
                result['final_classification'] = '⚠️ Uncertain - Potential Spam'
                result['risk_level'] = 'Medium'
                result['advice'] = (
                    'Our classifiers disagree on this email. '
                    'Treat with caution and verify the sender before taking any action.'
                )
    else:
        # No conflict - normal classification
        if top_pred in PHISHING_DESCRIPTIONS:
            info = PHISHING_DESCRIPTIONS[top_pred]
            result['final_classification'] = info['label']
            result['risk_level'] = info['risk_level']
            result['advice'] = info['advice']
        else:
            result['final_classification'] = f'⚠️ {top_pred}'
            result['risk_level'] = 'Unknown'
            result['advice'] = 'Be cautious with this email.'
        
        # Add low confidence warning even when no conflict
        if top_confidence < 0.5:
            result['uncertainty_warning'] = (
                f"⚠️ LOW CONFIDENCE: The phishing type classification confidence is only "
                f"{top_confidence:.1%}. The actual threat type may differ."
            )
            result['risk_level'] = 'Medium' if result['risk_level'] == 'Low' else result['risk_level']
    
    return result


def display_result(result):
    """
    Display classification results in a user-friendly format.
    Shows conflict warnings and uncertainty information when applicable.
    """
    print("📧 EMAIL CLASSIFICATION RESULT")
    print()
    print("📝 Email Preview:")
    print(f"    {result['email_text']}")
    print()
    
    # Stage 1 Result
    print("🔍 Stage 1 - Spam Detection:")
    print(f"    Result:     {result['stage1_result']}")
    print(f"    Confidence: {result['stage1_confidence']:.1%}")
    print()
    
    # Stage 2 Result (if applicable)
    if result['stage2_result']:
        print("🎯 Stage 2 - Phishing Type Classification:")
        print(f"    Prediction: {result['stage2_result']}")
        print(f"    Confidence: {result['stage2_confidence']:.1%}")
        
        # Show alternative prediction if available
        if result.get('stage2_alternative'):
            print(f"    2nd Best:   {result['stage2_alternative']} ({result['stage2_alternative_confidence']:.1%})")
        print()
    
    # Show conflict indicator and warning if present
    if result.get('has_conflict'):
        print("🔀 CONFLICT DETECTED")
        print(f"    {result['uncertainty_warning']}")
        print()
    elif result.get('uncertainty_warning'):
        print("⚠️  WARNING")
        print(f"    {result['uncertainty_warning']}")
        print()
    
    # Final Result
    print("📊 FINAL CLASSIFICATION:")
    print(f"    Result:     {result['final_classification']}")
    print(f"    Risk Level: {result['risk_level']}")
    print()
    
    print("💡 ADVICE:")
    print(f"    {result['advice']}")
    print()


def classify_batch(emails, spam_model, spam_vectorizer, phishing_model, phishing_vectorizer):
    """
    Classify a batch of emails.
    
    Returns:
        pd.DataFrame: Results for all emails including conflict detection
    """
    results = []
    
    for email in emails:
        result = classify_email(email, spam_model, spam_vectorizer, phishing_model, phishing_vectorizer)
        results.append({
            'email_preview': result['email_text'][:100] + '...' if len(result['email_text']) > 100 else result['email_text'],
            'spam_detection': result['stage1_result'],
            'spam_confidence': f"{result['stage1_confidence']:.1%}",
            'phishing_type': result['stage2_result'] if result['stage2_result'] else 'N/A',
            'phishing_confidence': f"{result['stage2_confidence']:.1%}" if result['stage2_confidence'] else 'N/A',
            'alternative_type': result.get('stage2_alternative', 'N/A'),
            'alternative_confidence': f"{result['stage2_alternative_confidence']:.1%}" if result.get('stage2_alternative_confidence') else 'N/A',
            'has_conflict': '⚠️ Yes' if result.get('has_conflict') else 'No',
            'final_classification': result['final_classification'],
            'risk_level': result['risk_level']
        })
    
    return pd.DataFrame(results)
