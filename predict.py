import torch
from model import DomainTransformer
import pandas as pd

# Load the model and the token-to-id mapping
def load_model(model_path, token_to_id_path):
    token_to_id = pd.read_json(token_to_id_path, typ='series').to_dict()
    id_to_token = {id: token for token, id in token_to_id.items()}

    model = DomainTransformer(len(token_to_id))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, token_to_id, id_to_token

# Function to encode the input domain
def encode_input(domain, token_to_id, max_length=20):
    tokens = [token_to_id.get(char, token_to_id['<UNK>']) for char in domain]
    return torch.LongTensor(tokens[:max_length])

# Function to decode the model's output
def decode_output(output, id_to_token):
    decoded_chars = [id_to_token.get(token_id.item(), '') for token_id in output]
    return ''.join(decoded_chars)


# Function for prediction
def predict(model, input_domain, token_to_id, id_to_token, device):
    encoded_input = encode_input(input_domain, token_to_id).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded_output = model(encoded_input, encoded_input)

    output_tokens = encoded_output.argmax(dim=2)[0]

    decoded_output = decode_output(output_tokens, id_to_token)
    return decoded_output

# Main function for prediction
def main():
    model_path = 'model.pth'
    token_to_id_path = 'token_to_id.json'

    # Load model and mapping
    model, token_to_id, id_to_token = load_model(model_path, token_to_id_path)
    
    # Specify the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Example prediction
    input_domain = ['google.com', 'cloudnine.in','syrv', 'web-domain','women','man']
    for predict_domain in input_domain:
        predicted_domain = predict(model, predict_domain, token_to_id, id_to_token, device)
        print(f'Input: {predict_domain}\nPredicted: {predicted_domain}')

if __name__ == "__main__":
    main()
