{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2530b89e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-07-23 16:40:07.199233: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d90a52f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting protobuf==3.19.0\n",
      "  Downloading protobuf-3.19.0-cp39-cp39-macosx_10_9_x86_64.whl (1.0 MB)\n",
      "\u001b[K     |████████████████████████████████| 1.0 MB 2.4 MB/s eta 0:00:01\n",
      "\u001b[?25hInstalling collected packages: protobuf\n",
      "  Attempting uninstall: protobuf\n",
      "    Found existing installation: protobuf 4.23.4\n",
      "    Uninstalling protobuf-4.23.4:\n",
      "      Successfully uninstalled protobuf-4.23.4\n",
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "streamlit 1.25.0 requires protobuf<5,>=3.20, but you have protobuf 3.19.0 which is incompatible.\u001b[0m\n",
      "Successfully installed protobuf-3.19.0\n"
     ]
    }
   ],
   "source": [
    "!pip install protobuf==3.19.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2c13bf52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (1.25.0)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (2.8.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (13.4.2)\n",
      "Requirement already satisfied: validators<1,>=0.2 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (0.20.0)\n",
      "Requirement already satisfied: tzlocal<5,>=1.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (4.3.1)\n",
      "Requirement already satisfied: requests<3,>=2.18 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (2.27.1)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (4.2.2)\n",
      "Requirement already satisfied: pympler<2,>=0.9 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (1.0.1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (5.0.1)\n",
      "Collecting protobuf<5,>=3.20\n",
      "  Using cached protobuf-4.23.4-cp37-abi3-macosx_10_9_universal2.whl (400 kB)\n",
      "Requirement already satisfied: click<9,>=7.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (8.0.4)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (1.21.0)\n",
      "Requirement already satisfied: pyarrow>=6.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (12.0.1)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (21.3)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (3.1.32)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (1.4)\n",
      "Requirement already satisfied: pydeck<1,>=0.8 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (1.4.1)\n",
      "Requirement already satisfied: pillow<10,>=7.1.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (9.0.1)\n",
      "Requirement already satisfied: importlib-metadata<7,>=1.4 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (4.11.3)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (6.1)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.1.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (4.1.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toolz in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from altair<6,>=4.0->streamlit) (0.11.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from altair<6,>=4.0->streamlit) (3.2.0)\n",
      "Requirement already satisfied: jinja2 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from altair<6,>=4.0->streamlit) (2.11.3)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.10)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.0)\n",
      "Requirement already satisfied: zipp>=0.5 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from importlib-metadata<7,>=1.4->streamlit) (3.7.0)\n",
      "Requirement already satisfied: six>=1.11.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (1.16.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (21.4.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: setuptools in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (58.0.4)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from packaging<24,>=16.8->streamlit) (3.0.4)\n",
      "Requirement already satisfied: pytz>=2020.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from pandas<3,>=1.3.0->streamlit) (2021.3)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from jinja2->altair<6,>=4.0->streamlit) (1.1.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from requests<3,>=2.18->streamlit) (2022.12.7)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from requests<3,>=2.18->streamlit) (1.26.8)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from requests<3,>=2.18->streamlit) (3.3)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from requests<3,>=2.18->streamlit) (2.0.4)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
      "Requirement already satisfied: pytz-deprecation-shim in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from tzlocal<5,>=1.1->streamlit) (0.1.0.post0)\n",
      "Requirement already satisfied: decorator>=3.4.0 in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from validators<1,>=0.2->streamlit) (5.1.1)\n",
      "Requirement already satisfied: tzdata in /usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages (from pytz-deprecation-shim->tzlocal<5,>=1.1->streamlit) (2023.3)\n",
      "Installing collected packages: protobuf\n",
      "  Attempting uninstall: protobuf\n",
      "    Found existing installation: protobuf 3.19.0\n",
      "    Uninstalling protobuf-3.19.0:\n",
      "      Successfully uninstalled protobuf-3.19.0\n",
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "tensorflow 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 4.23.4 which is incompatible.\n",
      "tensorboard 2.11.2 requires protobuf<4,>=3.9.2, but you have protobuf 4.23.4 which is incompatible.\u001b[0m\n",
      "Successfully installed protobuf-4.23.4\n"
     ]
    }
   ],
   "source": [
    "!pip install --upgrade streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3f87aa98",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "      👋 \u001b[1mWelcome to Streamlit!\u001b[0m\n",
      "\n",
      "      If you’d like to receive helpful onboarding emails, news, offers, promotions,\n",
      "      and the occasional swag, please enter your email address below. Otherwise,\n",
      "      leave this field blank.\n",
      "\n",
      "      \u001b[34mEmail: \u001b[0m ^C\n",
      "2023-07-23 16:58:31.758 \n"
     ]
    }
   ],
   "source": [
    "!streamlit hello"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a382e98c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streamlit, version 1.25.0\r\n"
     ]
    }
   ],
   "source": [
    "!streamlit --version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "543d86da",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define constants for dataset preparation and model training\n",
    "NUM_FRAMES = 30\n",
    "IMG_HEIGHT = 128\n",
    "IMG_WIDTH = 128"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eeb7bae8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to extract frames from a video file\n",
    "def extract_frames(video_path):\n",
    "    frames = []\n",
    "    cap = cv2.VideoCapture(video_path)\n",
    "    frame_count = 0\n",
    "    while True:\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            break\n",
    "        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))\n",
    "        frames.append(frame)\n",
    "        frame_count += 1\n",
    "        if frame_count == NUM_FRAMES:\n",
    "            break\n",
    "    cap.release()\n",
    "    if len(frames) < NUM_FRAMES:\n",
    "        padding = [np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))] * (NUM_FRAMES - len(frames))\n",
    "        frames.extend(padding)\n",
    "    return frames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4fa8d8ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to prepare the dataset for training and testing\n",
    "def prepare_dataset(fight_dir, non_fight_dir):\n",
    "    fight_frames = []\n",
    "    non_fight_frames = []\n",
    "    for file in os.listdir(fight_dir):\n",
    "        file_path = os.path.join(fight_dir, file)\n",
    "        frames = extract_frames(file_path)\n",
    "        fight_frames.extend(frames)\n",
    "    for file in os.listdir(non_fight_dir):\n",
    "        file_path = os.path.join(non_fight_dir, file)\n",
    "        frames = extract_frames(file_path)\n",
    "        non_fight_frames.extend(frames)\n",
    "    X = np.array(fight_frames + non_fight_frames)\n",
    "    y = np.concatenate([np.ones(len(fight_frames)), np.zeros(len(non_fight_frames))])\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
    "    X_train = X_train.reshape(X_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 3)\n",
    "    X_test = X_test.reshape(X_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 3)\n",
    "    X_train = X_train.astype('float32') / 255\n",
    "    X_test = X_test.astype('float32') / 255\n",
    "    return X_train, X_test, y_train, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5d395c6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-07-23 17:05:08.973610: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    }
   ],
   "source": [
    "# Define the CNN model architecture\n",
    "model = Sequential()\n",
    "model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(128, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(1, activation='sigmoid'))\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "81c07292",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare the dataset for training and testing\n",
    "#/Users/elijahadedamola/Downloads/(P-009831)_Artefact/data/fights\n",
    "fight_dir = r'/Users/elijahadedamola/Downloads/(P-009831)_Artefact/data/fights'\n",
    "non_fight_dir = r'/Users/elijahadedamola/Downloads/(P-009831)_Artefact/data/noFights'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "75a364ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = prepare_dataset(fight_dir, non_fight_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "97ab1c78",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "901/901 [==============================] - 1314s 1s/step - loss: 0.1040 - accuracy: 0.9611 - val_loss: 0.0257 - val_accuracy: 0.9903\n",
      "Epoch 2/5\n",
      "901/901 [==============================] - 606s 671ms/step - loss: 0.0204 - accuracy: 0.9931 - val_loss: 0.0100 - val_accuracy: 0.9972\n",
      "Epoch 3/5\n",
      "901/901 [==============================] - 658s 729ms/step - loss: 0.0152 - accuracy: 0.9954 - val_loss: 0.0201 - val_accuracy: 0.9925\n",
      "Epoch 4/5\n",
      "901/901 [==============================] - 1382s 2s/step - loss: 0.0088 - accuracy: 0.9970 - val_loss: 0.0540 - val_accuracy: 0.9878\n",
      "Epoch 5/5\n",
      "901/901 [==============================] - 1383s 2s/step - loss: 0.0070 - accuracy: 0.9978 - val_loss: 0.0028 - val_accuracy: 0.9989\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train,y_train,epochs=5, batch_size=32, validation_data=(X_test,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5922b539",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'Sequential' object has no attribute 'plot'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/k8/kj0hpdgn1_x3x8zdp4pzlqm80000gn/T/ipykernel_35062/2119193627.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mmodel_performance\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhistory\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m: 'Sequential' object has no attribute 'plot'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "model_performance = pd.DataFrame(history.history)\n",
    "model.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "77b352b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "226/226 [==============================] - 78s 339ms/step\n"
     ]
    }
   ],
   "source": [
    "predictions = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "e6b953f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "94bd786f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns \n",
    "sns.set_style('whitegrid')\n",
    "sns.set_context('notebook')\n",
    "\n",
    "_ = sns.heatmap(confusion_matrix(y_test,predictions.argmax(axis=1)), cmap=\"Blues_r\", lw=.5, annot=True, fmt=\".2f\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "88c4feb7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.51      1.00      0.68      3686\n",
      "         1.0       0.00      0.00      0.00      3520\n",
      "\n",
      "    accuracy                           0.51      7206\n",
      "   macro avg       0.26      0.50      0.34      7206\n",
      "weighted avg       0.26      0.51      0.35      7206\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/Caskroom/miniforge/base/envs/py3k/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,predictions.argmax(axis=1)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e11849ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"MyModel.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "df22d161",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/bin/bash: -c: line 0: syntax error near unexpected token `('\r\n",
      "/bin/bash: -c: line 0: `streamlit run Downloads/(P-009831)_Artefact/code1.py'\r\n"
     ]
    }
   ],
   "source": [
    "!streamlit run Downloads/(P-009831)_Artefact/code1.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8daff5e2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py3k",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
