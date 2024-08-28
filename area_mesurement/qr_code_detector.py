import cv2

# Carregar a imagem
image = cv2.imread('area_mesurement/image_qrcode.jpeg')

# Converter a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Inicializar o detector de QRCode
detector = cv2.QRCodeDetector()

# Detectar e decodificar o QR Code
data, vertices, _ = detector.detectAndDecode(gray)

# Verificar se o QR Code foi detectado
if vertices is not None:
    # Obter os pontos dos vértices
    points = vertices[0].astype(int)
    
    # Calcular a área do QR Code
    area = cv2.contourArea(points)
    
    # Desenhar o contorno do QR Code na imagem original
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=5)
    
    # Mostrar a área calculada
    print(f"Área do QR Code: {area} pixels quadrados")
    
    # Exibir a imagem com o QR Code detectado
    cv2.imshow('QR Code Detectado', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("QR Code não detectado.")

