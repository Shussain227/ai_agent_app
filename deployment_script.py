#!/usr/bin/env python3
"""
Deployment script for AI Agents Demo on Streamlit Cloud
This script helps prepare and validate your deployment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".streamlit/config.toml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def validate_requirements():
    """Validate requirements.txt"""
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip().split("\n")
        
        required_packages = [
            "streamlit",
            "openai", 
            "chromadb",
            "tiktoken"
        ]
        
        missing_packages = []
        for package in required_packages:
            if not any(package in req for req in requirements):
                missing_packages.append(package)
        
        if missing_packages:
            print("❌ Missing required packages in requirements.txt:")
            for package in missing_packages:
                print(f"   - {package}")
            return False
        
        print("✅ All required packages in requirements.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_streamlit_cloud_ready():
    """Check if the app is ready for Streamlit Cloud deployment"""
    print("\n🚀 Checking Streamlit Cloud readiness...\n")
    
    checks = [
        ("Required files", check_requirements),
        ("Requirements validation", validate_requirements),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
        print()
    
    return all_passed

def create_deployment_info():
    """Create deployment information file"""
    info = {
        "app_name": "AI Agents Demo",
        "framework": "Streamlit",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "main_file": "app.py",
        "required_secrets": [
            "OPENAI_API_KEY (optional - can be entered in UI)"
        ],
        "deployment_url": "https://share.streamlit.io/",
        "repository_structure": {
            "app.py": "Main Streamlit application",
            "requirements.txt": "Python dependencies",
            "README.md": "Documentation and setup guide",
            ".streamlit/config.toml": "Streamlit configuration",
            "deploy.py": "This deployment script"
        }
    }
    
    with open("deployment_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("✅ Created deployment_info.json")

def print_deployment_instructions():
    """Print step-by-step deployment instructions"""
    print("\n" + "="*60)
    print("🎉 STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n1. 📋 PREPARE YOUR REPOSITORY:")
    print("   • Push all files to your GitHub repository")
    print("   • Ensure your repository is public")
    print("   • Main branch should contain all files")
    
    print("\n2. 🌐 DEPLOY ON STREAMLIT CLOUD:")
    print("   • Go to: https://share.streamlit.io/")
    print("   • Click 'New app'")
    print("   • Connect your GitHub repository")
    print("   • Set main file: app.py")
    print("   • Deploy!")
    
    print("\n3. 🔑 CONFIGURE API KEY (Optional):")
    print("   • In Streamlit Cloud dashboard, go to app settings")
    print("   • Add secret: OPENAI_API_KEY = 'your-key-here'")
    print("   • Or users can enter it directly in the app")
    
    print("\n4. 🎯 TEST YOUR DEPLOYMENT:")
    print("   • Wait for deployment to complete")
    print("   • Test both Simple Task Agent and RAG Agent")
    print("   • Upload test documents to RAG agent")
    print("   • Verify all functionality works")
    
    print("\n" + "="*60)
    print("🚀 Your AI Agents Demo is ready to deploy!")
    print("="*60)

def main():
    print("🤖 AI Agents Demo - Deployment Preparation\n")
    
    if check_streamlit_cloud_ready():
        create_deployment_info()
        print_deployment_instructions()
        
        print("\n💡 QUICK DEPLOY LINKS:")
        print("• Streamlit Cloud: https://share.streamlit.io/")
        print("• Documentation: https://docs.streamlit.io/streamlit-cloud")
        
        return 0
    else:
        print("\n❌ Deployment preparation failed!")
        print("Please fix the issues above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())