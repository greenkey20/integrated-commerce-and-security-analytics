"""
텍스트 분석 전용 딥러닝 모델
기존 패턴을 따르되 텍스트 특화 기능 추가
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import streamlit as st
from config.settings import TextAnalyticsConfig


class TextAnalyticsModels:
    """텍스트 분석 전용 딥러닝 모델 클래스"""

    def __init__(self):
        self.sentiment_model = None
        self.classification_model = None
        self.tokenizer = None

    def create_sentiment_lstm(self, vocab_size=None, embedding_dim=None, lstm_units=None):
        """감정 분석용 LSTM 모델 생성"""

        # 기본값 설정
        vocab_size = vocab_size or TextAnalyticsConfig.IMDB_VOCAB_SIZE
        embedding_dim = embedding_dim or TextAnalyticsConfig.IMDB_EMBEDDING_DIM
        lstm_units = lstm_units or TextAnalyticsConfig.IMDB_LSTM_UNITS
        max_length = TextAnalyticsConfig.IMDB_MAX_LENGTH

        try:
            keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)

            import time
            timestamp = str(int(time.time() * 1000000))[-8:]

            model = keras.Sequential([
                layers.Embedding(
                    vocab_size, embedding_dim,
                    input_length=max_length,
                    name=f"sentiment_embedding_{timestamp}"
                ),
                layers.LSTM(
                    lstm_units,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    name=f"sentiment_lstm_{timestamp}"
                ),
                layers.Dropout(0.5, name=f"sentiment_dropout_{timestamp}"),
                layers.Dense(
                    1, activation='sigmoid',
                    name=f"sentiment_output_{timestamp}"
                )
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            self.sentiment_model = model
            return model, None

        except Exception as e:
            return None, f"감정 분석 모델 생성 실패: {str(e)}"

    def create_text_classifier(self, num_classes, vocab_size=None):
        """다중 클래스 텍스트 분류 모델 생성"""

        vocab_size = vocab_size or TextAnalyticsConfig.IMDB_VOCAB_SIZE

        try:
            keras.backend.clear_session()

            import time
            timestamp = str(int(time.time() * 1000000))[-8:]

            model = keras.Sequential([
                layers.Embedding(vocab_size, 128, name=f"class_embedding_{timestamp}"),
                layers.GlobalAveragePooling1D(name=f"class_pooling_{timestamp}"),
                layers.Dense(64, activation='relu', name=f"class_hidden_{timestamp}"),
                layers.Dropout(0.3, name=f"class_dropout_{timestamp}"),
                layers.Dense(num_classes, activation='softmax', name=f"class_output_{timestamp}")
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.classification_model = model
            return model, None

        except Exception as e:
            return None, f"텍스트 분류 모델 생성 실패: {str(e)}"

    def train_with_progress(self, model, X_train, y_train, X_val, y_val,
                            epochs=None, progress_bar=None, status_text=None):
        """진행상황을 표시하며 모델 훈련"""

        epochs = epochs or TextAnalyticsConfig.EPOCHS

        try:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=TextAnalyticsConfig.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True
                )
            ]

            # 기존 패턴과 동일한 Progress Callback
            if progress_bar and status_text:
                class TextProgressCallback(keras.callbacks.Callback):
                    def __init__(self, total_epochs, progress_bar, status_text):
                        super().__init__()
                        self.total_epochs = total_epochs
                        self.progress_bar = progress_bar
                        self.status_text = status_text

                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / self.total_epochs
                        self.progress_bar.progress(progress)

                        if logs:
                            self.status_text.text(
                                f"에포크 {epoch + 1}/{self.total_epochs} - "
                                f"손실: {logs.get('loss', 0):.4f}, "
                                f"정확도: {logs.get('accuracy', 0):.4f}, "
                                f"검증 정확도: {logs.get('val_accuracy', 0):.4f}"
                            )

                progress_callback = TextProgressCallback(epochs, progress_bar, status_text)
                callbacks.append(progress_callback)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=TextAnalyticsConfig.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            return history, None

        except Exception as e:
            return None, f"모델 훈련 실패: {str(e)}"