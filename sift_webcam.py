
import cv2
import numpy as np

def sift_webcam_matching():
    """
    Realiza correspondência SIFT em tempo real entre duas webcams.
    Pressione 'q' para sair.
    """
    # Inicializar as duas webcams
    cap1 = cv2.VideoCapture(0)  # Webcam 1
    cap2 = cv2.VideoCapture(2)  # Webcam 2

    if not cap1.isOpened() or not cap2.isOpened():
        print("Erro: Não foi possível abrir as webcams.")
        print("Verifique se duas webcams estão conectadas.")
        # Fallback: usar a mesma webcam para demonstração
        if cap1.isOpened():
            print("Usando webcam 1 para ambos os feeds (modo demonstração).")
            cap2 = cap1
        else:
            return

    # Criar detector SIFT
    sift = cv2.SIFT_create()

    # Configurar FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    print("Pressione 'q' para sair.")
    print("Pressione 's' para salvar um frame.")

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Erro ao capturar frame.")
            break

        # Redimensionar para melhorar performance
        frame1 = cv2.resize(frame1, (480, 360))
        frame2 = cv2.resize(frame2, (480, 360))

        # Converter para escala de cinza
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detectar keypoints e descritores SIFT
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is not None and des2 is not None and len(des1) > 2 and len(des2) > 2:
            # Encontrar correspondências
            matches = flann.knnMatch(des1, des2, k=2)

            # Teste de razão de Lowe
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Desenhar correspondências
            MIN_MATCH_COUNT = 10

            if len(good_matches) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    matchesMask = mask.ravel().tolist()

                    # Desenhar contorno projetado
                    h, w = gray1.shape
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
                    frame2_display = frame2.copy()
                    cv2.polylines(frame2_display, [np.int32(dst)], True, (0,255,0), 2)
                else:
                    matchesMask = None
                    frame2_display = frame2
            else:
                matchesMask = None
                frame2_display = frame2

            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                matchesMask=matchesMask,
                flags=0
            )

            result = cv2.drawMatches(frame1, kp1, frame2_display, kp2, 
                                     good_matches, None, **draw_params)

            # Adicionar informações na tela
            info_text = f"Matches: {len(good_matches)} | KP1: {len(kp1)} | KP2: {len(kp2)}"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        else:
            result = np.hstack([frame1, frame2])
            cv2.putText(result, "Sem features detectadas", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("SIFT Matching - Webcams", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"imagens/webcam_match_{frame_count}.png", result)
            print(f"Frame salvo: imagens/webcam_match_{frame_count}.png")
            frame_count += 1

    cap1.release()
    if cap2 != cap1:
        cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sift_webcam_matching()
