# Sequential Text-to-Motion Generation

## Project Summary
ParCo(Part-Coordinating Text-to-Motion Syenthesis)는 text로부터 사람의 동작을 생성하는 모델이다. 동작을 6개의 part(오른팔, 왼팔, 오른쪽 다리, 왼쪽 다리, 척추, 중심)로 나누고 각 part를 별도의 transformer를 통해 생성하며, Part Coordination module을 통해 전체 motion을 조율하여 통합한다.

ParCo는 "raise left hand and kick the door with right foot"과 같이 문장에 순차적인 흐름이 주어지면 행동 간 시간적 순서를 제대로 반영하지 못하는 한계를 보였다.

본 프로젝트는 이러한 ParCo의 문장 속 동작의 시간적 흐름을 제대로 반영하지 못하는 한계를 개선하는 것을 목표로 했다. 다음과 같은 실험을 통해 문장에 담긴 동작의 순서를 올바르게 반영하고, 동작의 자연스러움을 향상시키고자 하였다.

- VQ-VAE 구조에 positional embedding을 적용해 동작의 시간 흐름을 encoding하도록 개선하였다.
- CLIP의 token-level 출력과 GRU 출력을 연결(concatenate)하여 text embedding을 구성하였다.


## Code Instruction

## Demo

## Conclusion and Future Work
