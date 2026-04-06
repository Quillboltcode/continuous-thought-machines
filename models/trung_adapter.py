class CLIPAdapter(nn.Module):
    """CLIP-Adapter: Fine-tuning CLIP with a visual adapter"""

    def __init__(self, model_name, classnames, alpha=0.2, reduction=4):
        super().__init__()
        print(
            f"Initializing CLIP-Adapter with model: {model_name}, alpha: {alpha}, reduction: {reduction}"
        )
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        self.clip_model.float()
        verify_clip_architecture(self.clip_model)
        self.clip_model.eval()
        self.dtype = self.clip_model.dtype
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224).to(device)
            image_features = self.clip_model.encode_image(dummy_image.type(self.dtype))
            self.image_feature_dim = image_features.shape[-1]
        print(f"Image feature dimension: {self.image_feature_dim}")
        print(f"Adapter bottleneck dimension: {self.image_feature_dim // reduction}")
        self.dtype = torch.float32
        self.adapter = Adapter(self.image_feature_dim, reduction).to(
            device=device, dtype=self.dtype
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.alpha = alpha
        self.classnames = classnames
        self.emotion_descriptions = get_emotion_descriptions()
        self.encode_text_features()
        print_model_summary(self)

    def encode_text_features(self):
        print("Encoding emotion descriptions...")
        self.text_features = {}
        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                text_inputs = clip.tokenize(descriptions).to(device)
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features[emotion] = text_features
            self.emotion_embedding_tensor = torch.cat(
                [self.text_features[emotion] for emotion in EMOTIONS], dim=0
            )
        print(f"Text embeddings shape: {self.emotion_embedding_tensor.shape}")

    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=1e-5):
        self.adapter.train()
        optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
        temperature = self.clip_model.logit_scale.exp().item()
        print(f"Using temperature: {temperature:.2f}")
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(num_epochs):
            total_train_loss = total_train_correct = total_train_samples = (
                batch_count
            ) = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
            )
            for pixel_values, labels, _ in progress_bar:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(
                        pixel_values.type(self.dtype)
                    )
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                adapter_output = self.adapter(image_features)
                final_image_features = (
                    self.alpha * adapter_output + (1 - self.alpha) * image_features
                )
                final_image_features = final_image_features / final_image_features.norm(
                    dim=-1, keepdim=True
                )
                with torch.no_grad():
                    text_features = self.emotion_embedding_tensor
                logits = temperature * torch.matmul(
                    final_image_features, text_features.T
                )
                loss = nn.CrossEntropyLoss()(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_train_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_train_samples += labels.size(0)
                batch_count += 1
                progress_bar.set_postfix(
                    {
                        "Train Loss": f"{total_train_loss / batch_count:.4f}",
                        "Train Acc": f"{total_train_correct / total_train_samples:.4f}",
                    }
                )
            avg_train_loss = total_train_loss / batch_count
            train_accuracy = total_train_correct / total_train_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            self.adapter.eval()
            total_val_loss = total_val_correct = total_val_samples = val_batch_count = 0
            val_progress_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
            )
            with torch.no_grad():
                for pixel_values, labels, _ in val_progress_bar:
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)
                    image_features = self.clip_model.encode_image(
                        pixel_values.type(self.dtype)
                    )
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    adapter_output = self.adapter(image_features)
                    final_image_features = (
                        self.alpha * adapter_output + (1 - self.alpha) * image_features
                    )
                    final_image_features = (
                        final_image_features
                        / final_image_features.norm(dim=-1, keepdim=True)
                    )
                    text_features = self.emotion_embedding_tensor
                    logits = temperature * torch.matmul(
                        final_image_features, text_features.T
                    )
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    total_val_loss += loss.item()
                    total_val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    total_val_samples += labels.size(0)
                    val_batch_count += 1
                    val_progress_bar.set_postfix(
                        {
                            "Val Loss": f"{total_val_loss / val_batch_count:.4f}",
                            "Val Acc": f"{total_val_correct / total_val_samples:.4f}",
                        }
                    )
            avg_val_loss = total_val_loss / val_batch_count
            val_accuracy = total_val_correct / total_val_samples
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.adapter.state_dict()
                torch.save(
                    best_model_state,
                    f"best_model_epoch_{epoch + 1}_val_loss_{best_val_loss:.4f}.pth",
                )
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
            )
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            self.adapter.train()

        if best_model_state is not None:
            self.adapter.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.4f}")

        self.adapter.eval()
        return train_losses, val_losses, train_accuracies, val_accuracies, best_val_loss

    def predict(self, pixel_values):
        """Predict emotion probabilities from image pixel values"""
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            image_features = self.clip_model.encode_image(pixel_values.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            adapter_output = self.adapter(image_features)
            final_image_features = (
                self.alpha * adapter_output + (1 - self.alpha) * image_features
            )
            final_image_features = final_image_features / final_image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = self.emotion_embedding_tensor
            similarity = 100 * torch.matmul(final_image_features, text_features.T)
            probs = torch.softmax(similarity, dim=1)
        return probs
