import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { PaperProvider } from 'react-native-paper';

import DashboardScreen from './src/screens/DashboardScreen';
import StudySetupScreen from './src/screens/StudySetupScreen';
import DataInputScreen from './src/screens/DataInputScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import { initDb } from './src/database/db';
import { theme } from './src/theme';

export type RootStackParamList = {
  Dashboard: undefined;
  StudySetup: undefined;
  DataInput: { studyId: string };
  Results: { studyId: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

// Initialize database on startup
initDb();

export default function App() {
  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Stack.Navigator 
          initialRouteName="Dashboard"
          screenOptions={{
            headerShown: false,
            contentStyle: { backgroundColor: theme.colors.background }
          }}
        >
          <Stack.Screen name="Dashboard" component={DashboardScreen} />
          <Stack.Screen name="StudySetup" component={StudySetupScreen} />
          <Stack.Screen name="DataInput" component={DataInputScreen} />
          <Stack.Screen name="Results" component={ResultsScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
