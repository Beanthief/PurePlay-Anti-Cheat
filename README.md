# PurePlay AntiCheat (WIP)
**Free, Open-Source AntiCheat for Competitive Game Developers**
- *Note: This project is in development and is currently acting as my college Capstone project. It is not a working product.*

## Project Overview
**PurePlay AntiCheat** is an easy-to-implement, open-source anticheat solution designed to help game developers detect inhuman inputs across GamePad, Mouse, and Keyboard. By leveraging trained neural networks, the API can identify input sources using external devices or software cheats such as Xim, Cronus, or custom Arduino-based setups.

This solution is perfect for **low-budget game developers** who need a reliable anticheat system that doesn’t require extensive networking integration or costly licensing. As it is only designed to catch external cheating sources, it is recommended to be used alongside an anticheat capable of catching memory-based cheats.

## Features
- 🎮 **Supports GamePad, Mouse, and Keyboard Inputs**
- 🤖 **AI-Based Detection**: Trained neural networks classify human vs inhuman inputs with over 80% confidence.
- 💾 **Cross-Engine Compatibility**: Runs externally, independent of the game engine.
- ⚙️ **Customizable Tuning**: Developers can configure client/server models.
- 💻 **Lightweight Client & Server Models**: Supports server-only, client-only, or both depending on your setup.
- 💡 **Open Source & Free to Use**: No licensing fees, fully transparent, and community-driven.

## Model Tradeoffs
1. **Client-Only Model**:
   - Requires careful configuration to minimize performance impact.
   - Client-side processing only; may need manual review or automated filtering for results.

2. **Server-Only Model**:
   - Requires AI acceleration hardware on servers, which may be costly but reduces client performance impacts.
   - Integration with the game engine required for networked view angles.

3. **Combined Model**:
   - Offers the highest accuracy.
   - Requires AI-capable hardware on both the server and client sides.
   - Involves deeper integration with the game engine.

## Getting Started
### Installation
To get started with **PurePlay AntiCheat**, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/Beanthief/PurePlay-Anti-Cheat
cd PurePlay-Anti-Cheat
pip install -r requirements.txt
